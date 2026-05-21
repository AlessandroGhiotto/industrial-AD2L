"""DRAEM: Reconstructive subnetwork + Discriminative subnetwork trained on
synthetic anomalies (fractal noise mask blended with texture images).

Reference: Zavrtanik et al., "DRAEM - A discriminatively trained reconstruction
embedding for surface anomaly detection", ICCV 2021.

Differences from the paper for simplicity:
  - Uses multi-octave bicubic-upsampled Gaussian noise as the mask source
    instead of true Perlin noise. Same character (blob-like binary masks at
    multiple scales), much simpler implementation.
  - Reconstruction loss is L2 only (paper uses L2 + SSIM). SSIM can be added
    later; L2 is enough to get the discriminator working.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- synthetic anomaly mask ----------

def _fractal_noise(H, W, n_octaves, base_scale, rng):
    total = np.zeros((H, W), dtype=np.float32)
    amp = 1.0
    for o in range(n_octaves):
        s = max(2, int(base_scale * (2 ** o)))
        low = rng.standard_normal((s, s)).astype(np.float32)
        low_t = torch.from_numpy(low).unsqueeze(0).unsqueeze(0)
        up = F.interpolate(low_t, size=(H, W), mode="bicubic", align_corners=False)
        total += amp * up.squeeze().numpy()
        amp *= 0.5
    std = total.std()
    if std < 1e-8:
        return total
    return (total - total.mean()) / std


def generate_anomaly_mask(H, W, threshold=0.5, n_octaves=3, base_scale_range=(4, 16), rng=None):
    """Random binary mask via multi-octave fractal noise + thresholding.

    Returns float32 array of shape (H, W) in {0, 1}.
    """
    if rng is None:
        rng = np.random.default_rng()
    base_scale = int(rng.integers(base_scale_range[0], base_scale_range[1] + 1))
    noise = _fractal_noise(H, W, n_octaves=n_octaves, base_scale=base_scale, rng=rng)
    return (noise > threshold).astype(np.float32)


# ---------- synthetic anomaly generator (image + texture -> augmented + mask) ----------

class DraemAnomalyGenerator:
    """Blend texture image into clean image inside a random fractal-noise mask.

    DRAEM augmentation formula:
        I_aug = I + beta * M * (A - I)
    where M is a binary mask and beta ~ U(beta_low, beta_high).

    With probability (1 - p_aug) the clean image is returned unchanged (mask=0).
    """

    def __init__(self, mask_threshold=0.5, n_octaves=3, base_scale_range=(4, 16),
                 beta_low=0.1, beta_high=1.0, p_aug=0.5, seed=None):
        self.mask_threshold = mask_threshold
        self.n_octaves = n_octaves
        self.base_scale_range = base_scale_range
        self.beta_low = beta_low
        self.beta_high = beta_high
        self.p_aug = p_aug
        self.rng = np.random.default_rng(seed)

    def __call__(self, images, textures):
        """images, textures: [B, 3, H, W] float tensors in [0, 1].
        Returns (augmented_images, masks) on the same device.
        """
        device = images.device
        imgs_np = images.detach().cpu().numpy()
        tex_np = textures.detach().cpu().numpy()
        B, C, H, W = imgs_np.shape
        aug = imgs_np.copy()
        mask = np.zeros((B, 1, H, W), dtype=np.float32)
        for i in range(B):
            if self.rng.random() > self.p_aug:
                continue
            m = generate_anomaly_mask(
                H, W,
                threshold=self.mask_threshold,
                n_octaves=self.n_octaves,
                base_scale_range=self.base_scale_range,
                rng=self.rng,
            )
            beta = float(self.rng.uniform(self.beta_low, self.beta_high))
            m3 = m[None]  # [1, H, W] broadcasts to channels
            blended = imgs_np[i] + beta * m3 * (tex_np[i] - imgs_np[i])
            aug[i] = np.clip(blended, 0.0, 1.0)
            mask[i, 0] = m
        return (
            torch.from_numpy(aug).to(device),
            torch.from_numpy(mask).to(device),
        )


# ---------- losses ----------

class FocalLoss(nn.Module):
    """Focal loss over 2-channel logits and {0, 1} target masks.

    Inputs: logits [B, 2, H, W], target [B, H, W] in {0, 1}.
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=1)
        target_long = target.long().unsqueeze(1)
        log_p_t = log_p.gather(1, target_long).squeeze(1)
        p_t = log_p_t.exp()
        loss = -((1 - p_t) ** self.gamma) * log_p_t
        return loss.mean()


# ---------- U-Net blocks ----------

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _upsample(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class ReconstructiveSubnetwork(nn.Module):
    """Encoder-decoder WITHOUT skip connections.

    Paper rationale: skip connections would let anomalous appearance leak
    through to the reconstruction, defeating the purpose. The model must
    reconstruct the clean image from a (possibly corrupted) input using only
    the bottleneck representation.
    """

    def __init__(self, in_channels=3, out_channels=3, base=32):
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.enc4 = _ConvBlock(base * 4, base * 8)
        self.bottleneck = _ConvBlock(base * 8, base * 8)
        self.dec4 = _ConvBlock(base * 8, base * 8)
        self.dec3 = _ConvBlock(base * 8, base * 4)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(_upsample(b, e4))
        d3 = self.dec3(_upsample(d4, e3))
        d2 = self.dec2(_upsample(d3, e2))
        d1 = self.dec1(_upsample(d2, e1))
        return self.out(d1)


class DiscriminativeSubnetwork(nn.Module):
    """U-Net WITH skip connections. Input is concat([orig, recon]) along channels."""

    def __init__(self, in_channels=6, out_channels=2, base=32):
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.enc4 = _ConvBlock(base * 4, base * 8)
        self.bottleneck = _ConvBlock(base * 8, base * 8)
        self.dec4 = _ConvBlock(base * 8 + base * 8, base * 8)
        self.dec3 = _ConvBlock(base * 8 + base * 4, base * 4)
        self.dec2 = _ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = _ConvBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([_upsample(b, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([_upsample(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([_upsample(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([_upsample(d2, e1), e1], dim=1))
        return self.out(d1)


# ---------- top-level DRAEM ----------

class DRAEM(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.reconstructive = ReconstructiveSubnetwork(3, 3, base=base)
        self.discriminative = DiscriminativeSubnetwork(6, 2, base=base)

    def _both(self, x):
        recon = self.reconstructive(x)
        joined = torch.cat([recon, x], dim=1)
        logits = self.discriminative(joined)
        return recon, logits

    def forward(self, x):
        recon, logits = self._both(x)
        if self.training:
            return recon, logits
        prob = F.softmax(logits, dim=1)[:, 1:2]
        return prob

    def anomaly_prob(self, x):
        """Always returns per-pixel anomaly probability [B, 1, H, W],
        regardless of self.training. Used for the auxiliary supervised loss."""
        _, logits = self._both(x)
        return F.softmax(logits, dim=1)[:, 1:2]
