import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SimpleNetAD(nn.Module):
    def __init__(
        self,
        backbone_name="wide_resnet50_2",
        adapter_dim=1024,
        noise_std=0.015,
        imagenet_normalize=True,
    ):
        """
        SimpleNet for Anomaly Detection with Multi-Layer Fusion.

        Args:
            backbone_name: Name of the timm backbone.
            adapter_dim: Dimension of the adapted feature space.
            noise_std: Multiplier applied to per-channel feature std when sampling
                synthetic anomalies (so noise magnitude matches the local feature scale).
            imagenet_normalize: If True, apply ImageNet mean/std normalization inside
                forward. The dataset returns raw [0, 1] tensors, so this is required when
                the backbone is pretrained on ImageNet.
        """
        super().__init__()
        self.noise_std = noise_std
        self.imagenet_normalize = imagenet_normalize

        # ImageNet normalization buffers (applied inside forward so any external code
        # that feeds raw [0, 1] tensors stays compatible).
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # 1. Feature Extractor (timm based)
        # out_indices=(1, 2, 3) corresponds to layers with strides 4, 8, 16
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3),
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Dynamic calculation of input dimension for adapter
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            if self.imagenet_normalize:
                dummy_input = (dummy_input - self.img_mean) / self.img_std
            features = self.backbone(dummy_input)
            in_dim = sum([f.shape[1] for f in features])

        # 2. Feature Adapter
        self.adapter = nn.Sequential(
            nn.Conv2d(in_dim, adapter_dim, kernel_size=1),
            nn.BatchNorm2d(adapter_dim),
            nn.ReLU(inplace=True),
        )

        # 3. Discriminator (MLP-like with 1x1 convs)
        self.discriminator = nn.Sequential(
            nn.Conv2d(adapter_dim, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def _extract_adapted_features(self, x):
        if self.imagenet_normalize:
            x = (x - self.img_mean) / self.img_std

        with torch.no_grad():
            self.backbone.eval()
            features = self.backbone(x)

        f1, f2, f3 = features
        target_size = f1.shape[2:]
        f2_up = F.interpolate(f2, size=target_size, mode="bilinear", align_corners=False)
        f3_up = F.interpolate(f3, size=target_size, mode="bilinear", align_corners=False)
        fused_features = torch.cat([f1, f2_up, f3_up], dim=1)

        return self.adapter(fused_features)

    def supervised_logits(self, x):
        """Per-pixel discriminator logits at feature-map resolution, no fake-noise path.

        Used for the auxiliary supervised loss on labeled anomaly masks.
        """
        adapted = self._extract_adapted_features(x)
        return self.discriminator(adapted)

    def forward(self, x):
        adapted_features = self._extract_adapted_features(x)

        if self.training:
            # Scale noise to per-channel feature std so synthetic anomalies are
            # uniformly meaningful across channels regardless of magnitude.
            with torch.no_grad():
                per_ch_std = adapted_features.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
            noise = torch.randn_like(adapted_features) * self.noise_std * per_ch_std
            fake_features = adapted_features + noise

            combined_features = torch.cat([adapted_features, fake_features], dim=0)
            logits = self.discriminator(combined_features)

            B = adapted_features.size(0)
            labels = torch.cat(
                [
                    torch.zeros(B, 1, logits.size(2), logits.size(3), device=x.device),
                    torch.ones(B, 1, logits.size(2), logits.size(3), device=x.device),
                ],
                dim=0,
            )
            return logits, labels

        # Inference
        logits = self.discriminator(adapted_features)
        anomaly_map = F.interpolate(
            logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return anomaly_map
