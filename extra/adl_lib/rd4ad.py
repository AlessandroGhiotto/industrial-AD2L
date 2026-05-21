"""
Reverse Distillation for Anomaly Detection (RD4AD)

Based on: "Anomaly Detection via Reverse Distillation from One-Class Embedding"
(Deng & Li, CVPR 2022)

Architecture:
    - Frozen pretrained encoder (teacher) extracts multi-scale features
    - One-Class Bottleneck Embedding (OCBE) compresses deepest features
    - Symmetric decoder reconstructs multi-scale features
    - Anomaly = cosine distance between teacher and decoder at each scale
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm

from adl_lib.config import (
    DEVICE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEED,
    RD4AD_BACKBONE,
    RD4AD_OUT_INDICES,
    RD4AD_BOTTLENECK_DIM,
    RD4AD_LR,
    RD4AD_WEIGHT_DECAY,
    RD4AD_EPOCHS,
    RD4AD_EARLY_STOP_PATIENCE,
    RD4AD_ANOMALY_MAP_SIGMA,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)
from adl_lib.data import tensor_to_numpy_image, tensor_to_numpy_mask
from adl_lib.utils import postprocess_anomaly_map


# ===== Encoder (Frozen Teacher) =====


class FrozenEncoder(nn.Module):
    """Frozen pretrained backbone for multi-scale feature extraction."""

    def __init__(self, backbone_name=RD4AD_BACKBONE, out_indices=RD4AD_OUT_INDICES):
        super().__init__()
        self.backbone_name = backbone_name
        self.out_indices = tuple(out_indices)

        self.model = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=self.out_indices,
        ).eval()

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        # Determine channel dimensions via a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            normed = (dummy - self.mean) / self.std
            feats = self.model(normed)
            self.channels = [f.shape[1] for f in feats]
            self.spatial_sizes = [f.shape[2:] for f in feats]

        print(
            f"RD4AD Teacher: {backbone_name}, "
            f"layers={self.out_indices}, "
            f"channels={self.channels}, "
            f"spatial={[tuple(s) for s in self.spatial_sizes]}"
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


# ===== One-Class Bottleneck Embedding =====


class OCBE(nn.Module):
    """
    One-Class Bottleneck Embedding.
    Compresses the deepest encoder features into a compact representation
    that captures the one-class distribution of normal data.
    """

    def __init__(self, in_channels, bottleneck_dim=RD4AD_BOTTLENECK_DIM):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.bottleneck(x)


# ===== Multi-Scale Decoder =====


class DecoderBlock(nn.Module):
    """Single decoder block: upsample + conv to match target channels and spatial size."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2,
                padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleDecoder(nn.Module):
    """
    Symmetric decoder that reconstructs multi-scale features.
    Takes the bottleneck output (deepest level) and progressively upsamples
    to reconstruct features at each encoder level.
    """

    def __init__(self, encoder_channels):
        """
        Args:
            encoder_channels: list of channel dims from encoder, ordered shallow→deep
                e.g. [256, 512, 1024] for WRN50 layers (1,2,3)
        """
        super().__init__()
        # Decoder works from deepest to shallowest
        reversed_channels = list(reversed(encoder_channels))

        self.blocks = nn.ModuleList()
        for i in range(len(reversed_channels) - 1):
            self.blocks.append(
                DecoderBlock(reversed_channels[i], reversed_channels[i + 1])
            )

        # Output projection heads: one per scale to match teacher channels exactly
        self.output_heads = nn.ModuleList()
        for ch in encoder_channels:
            self.output_heads.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(ch),
                )
            )

    def forward(self, bottleneck_out, target_spatial_sizes):
        """
        Args:
            bottleneck_out: tensor from OCBE, same spatial/channel as deepest encoder level
            target_spatial_sizes: list of (H, W) for each encoder level, shallow→deep

        Returns:
            list of reconstructed features, ordered shallow→deep (matching encoder order)
        """
        x = bottleneck_out
        reversed_sizes = list(reversed(target_spatial_sizes))

        # Collect decoder outputs from deep to shallow
        reconstructed_reversed = [x]  # deepest level = bottleneck output directly
        for i, block in enumerate(self.blocks):
            x = block(x)
            target_h, target_w = reversed_sizes[i + 1]
            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            reconstructed_reversed.append(x)

        # Reverse to match encoder ordering (shallow→deep)
        reconstructed = list(reversed(reconstructed_reversed))

        # Apply output heads
        outputs = []
        for feat, head in zip(reconstructed, self.output_heads):
            outputs.append(head(feat))

        return outputs


# ===== Cosine Similarity Loss =====


def multi_scale_cosine_loss(teacher_features, decoder_features):
    """
    Compute cosine similarity loss between teacher and decoder features
    across all scales. Loss = 1 - cosine_similarity, averaged over scales.
    """
    total_loss = 0.0
    for t_feat, d_feat in zip(teacher_features, decoder_features):
        # Normalize along channel dimension
        t_norm = F.normalize(t_feat, p=2, dim=1)
        d_norm = F.normalize(d_feat, p=2, dim=1)
        # Cosine similarity per spatial location, averaged
        cos_sim = (t_norm * d_norm).sum(dim=1).mean()
        total_loss += (1.0 - cos_sim)
    return total_loss / len(teacher_features)


# ===== Anomaly Map Computation =====


def compute_anomaly_map(teacher_features, decoder_features, image_size=IMAGE_SIZE):
    """
    Compute pixel-level anomaly map as the aggregated cosine distance
    between teacher and decoder features across all scales.
    """
    anomaly_maps = []
    for t_feat, d_feat in zip(teacher_features, decoder_features):
        t_norm = F.normalize(t_feat, p=2, dim=1)
        d_norm = F.normalize(d_feat, p=2, dim=1)
        # Cosine distance per spatial location
        cos_dist = 1.0 - (t_norm * d_norm).sum(dim=1, keepdim=True)
        cos_dist = F.relu(cos_dist)  # Clamp to non-negative
        # Upsample to image size
        cos_dist = F.interpolate(
            cos_dist, size=(image_size, image_size),
            mode="bilinear", align_corners=False
        )
        anomaly_maps.append(cos_dist)

    # Sum across scales
    combined = sum(anomaly_maps) / len(anomaly_maps)
    return combined.squeeze(1)  # (B, H, W)


# ===== Main RD4AD Class =====


class RD4AD:
    """
    Reverse Distillation for Anomaly Detection.

    Usage:
        model = RD4AD()
        model.fit(train_loader, class_name="class_01")
        results = model.predict_labeled(val_loader)
        results = model.predict_unlabeled(test_loader)
    """

    def __init__(
        self,
        backbone_name=RD4AD_BACKBONE,
        out_indices=RD4AD_OUT_INDICES,
        bottleneck_dim=RD4AD_BOTTLENECK_DIM,
        lr=RD4AD_LR,
        weight_decay=RD4AD_WEIGHT_DECAY,
        epochs=RD4AD_EPOCHS,
        early_stop_patience=RD4AD_EARLY_STOP_PATIENCE,
        sigma=RD4AD_ANOMALY_MAP_SIGMA,
        checkpoint_dir="./checkpoints/rd4ad",
    ):
        self.backbone_name = backbone_name
        self.out_indices = tuple(out_indices)
        self.bottleneck_dim = bottleneck_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.sigma = sigma
        self.checkpoint_dir = Path(checkpoint_dir)
        self.class_name = None

        # Build encoder (frozen teacher)
        self.encoder = FrozenEncoder(backbone_name, out_indices).to(DEVICE).eval()

        # Build bottleneck and decoder (trainable)
        deepest_channels = self.encoder.channels[-1]
        self.bottleneck = OCBE(deepest_channels, bottleneck_dim).to(DEVICE)
        self.decoder = MultiScaleDecoder(self.encoder.channels).to(DEVICE)

        total_params = sum(
            p.numel() for p in list(self.bottleneck.parameters()) + list(self.decoder.parameters())
        )
        print(f"RD4AD trainable parameters: {total_params:,}")

    def _get_checkpoint_path(self, class_name):
        return self.checkpoint_dir / class_name / "rd4ad_checkpoint.pt"

    def _save_checkpoint(self, class_name, epoch=None, best_metric=None):
        """Save bottleneck + decoder weights."""
        ckpt_path = self._get_checkpoint_path(class_name)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "bottleneck": self.bottleneck.state_dict(),
            "decoder": self.decoder.state_dict(),
            "backbone_name": self.backbone_name,
            "out_indices": self.out_indices,
            "bottleneck_dim": self.bottleneck_dim,
            "epoch": epoch,
            "best_metric": best_metric,
        }
        torch.save(state, ckpt_path)
        print(f"RD4AD checkpoint saved: {ckpt_path}")

    def _load_checkpoint(self, class_name):
        """Load bottleneck + decoder weights if available."""
        ckpt_path = self._get_checkpoint_path(class_name)
        if not ckpt_path.exists():
            print(f"RD4AD checkpoint not found: {ckpt_path}")
            return False

        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        # Verify architecture compatibility
        if state.get("backbone_name") != self.backbone_name:
            print(
                f"Warning: backbone mismatch "
                f"(checkpoint={state.get('backbone_name')}, current={self.backbone_name})"
            )
            return False

        self.bottleneck.load_state_dict(state["bottleneck"])
        self.decoder.load_state_dict(state["decoder"])
        print(
            f"RD4AD checkpoint loaded: {ckpt_path} "
            f"(epoch={state.get('epoch')}, metric={state.get('best_metric', 'N/A')})"
        )
        return True

    def fit(self, train_loader, class_name="default", val_loader=None):
        """
        Train the bottleneck and decoder on normal images.

        Args:
            train_loader: DataLoader yielding dicts with 'image' key
            class_name: identifier for checkpoint saving
            val_loader: optional labeled anomaly loader for early stopping
        """
        self.class_name = class_name
        self.encoder.eval()
        self.bottleneck.train()
        self.decoder.train()

        optimizer = torch.optim.AdamW(
            list(self.bottleneck.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )

        best_metric = -1.0
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.bottleneck.train()
            self.decoder.train()

            epoch_losses = []
            for batch in tqdm(train_loader, desc=f"RD4AD epoch {epoch}/{self.epochs}", leave=False):
                images = batch["image"].to(DEVICE, non_blocking=True)

                with torch.no_grad():
                    teacher_features = self.encoder(images)

                # Forward through bottleneck and decoder
                bottleneck_out = self.bottleneck(teacher_features[-1])
                target_sizes = [f.shape[2:] for f in teacher_features]
                decoder_features = self.decoder(bottleneck_out, target_sizes)

                loss = multi_scale_cosine_loss(teacher_features, decoder_features)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_losses.append(float(loss.item()))

            scheduler.step()
            avg_loss = float(np.mean(epoch_losses))

            # Evaluate on validation set if available
            val_info = ""
            if val_loader is not None and epoch % 5 == 0:
                from adl_lib.utils import calibrate_threshold_from_labeled
                results = self.predict_labeled(val_loader)
                if len(results) > 0:
                    from sklearn.metrics import average_precision_score
                    pixel_masks = np.concatenate(
                        [r["mask"].reshape(-1) for r in results]
                    ).astype(np.uint8)
                    pixel_scores = np.concatenate(
                        [r["anomaly_map"].reshape(-1) for r in results]
                    ).astype(np.float32)
                    if np.unique(pixel_masks).size > 1:
                        pixel_ap = float(average_precision_score(pixel_masks, pixel_scores))
                        val_info = f" | pixel_AP={pixel_ap:.4f}"

                        if pixel_ap > best_metric:
                            best_metric = pixel_ap
                            patience_counter = 0
                            self._save_checkpoint(class_name, epoch, best_metric)
                        else:
                            patience_counter += 5  # We check every 5 epochs

            print(
                f"  Epoch {epoch:03d} | loss={avg_loss:.6f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}{val_info}"
            )

            # Early stopping
            if val_loader is not None and patience_counter >= self.early_stop_patience:
                print(f"  Early stopping at epoch {epoch} (patience={self.early_stop_patience})")
                # Reload best checkpoint
                self._load_checkpoint(class_name)
                break

        # Save final checkpoint if no validation was used or no improvement was saved
        if val_loader is None or best_metric < 0:
            self._save_checkpoint(class_name, self.epochs, None)

        return self

    @torch.no_grad()
    def _predict_batch(self, images):
        """Compute anomaly maps for a batch of images."""
        self.encoder.eval()
        self.bottleneck.eval()
        self.decoder.eval()

        images = images.to(DEVICE, non_blocking=True)
        teacher_features = self.encoder(images)
        bottleneck_out = self.bottleneck(teacher_features[-1])
        target_sizes = [f.shape[2:] for f in teacher_features]
        decoder_features = self.decoder(bottleneck_out, target_sizes)

        anomaly_maps = compute_anomaly_map(
            teacher_features, decoder_features, image_size=IMAGE_SIZE
        )
        return anomaly_maps

    @torch.no_grad()
    def predict_labeled(self, loader):
        """Predict on labeled anomaly data. Same interface as PatchCoreLite."""
        self.encoder.eval()
        self.bottleneck.eval()
        self.decoder.eval()

        results = []
        for batch in tqdm(loader, desc="RD4AD inference (labeled)"):
            images = batch["image"]
            anomaly_maps = self._predict_batch(images)

            b = images.shape[0]
            for i in range(b):
                amap = anomaly_maps[i].detach().cpu().numpy().astype(np.float32)
                amap = postprocess_anomaly_map(amap, sigma=self.sigma)
                image_score = float(amap.max())

                results.append({
                    "image": tensor_to_numpy_image(images[i]),
                    "anomaly_map": amap,
                    "mask": tensor_to_numpy_mask(batch["mask"][i]),
                    "label": int(batch["label"][i].item()),
                    "score": image_score,
                    "defect_type": batch["defect_type"][i],
                    "path": batch["path"][i],
                })

        return results

    @torch.no_grad()
    def predict_unlabeled(self, loader):
        """Predict on unlabeled test data. Same interface as PatchCoreLite."""
        self.encoder.eval()
        self.bottleneck.eval()
        self.decoder.eval()

        results = []
        for batch in tqdm(loader, desc="RD4AD inference (unlabeled)"):
            images = batch["image"]
            anomaly_maps = self._predict_batch(images)

            b = images.shape[0]
            for i in range(b):
                amap = anomaly_maps[i].detach().cpu().numpy().astype(np.float32)
                amap = postprocess_anomaly_map(amap, sigma=self.sigma)
                image_score = float(amap.max())

                results.append({
                    "image": tensor_to_numpy_image(images[i]),
                    "anomaly_map": amap,
                    "score": image_score,
                    "path": batch["path"][i],
                })

        return results
