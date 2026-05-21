import json
import random
import inspect
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
import timm

# ── GPU PCA ───────────────────────────────────────────────────────────────────

class GPUPCA:
    """Randomised-SVD PCA via torch.pca_lowrank"""
    def __init__(self, n_components, device='cuda', niter=4):
        self.n_components = n_components
        self.niter        = niter
        self.device       = device

    @torch.no_grad()
    def fit(self, X_np):
        X              = torch.from_numpy(X_np).to(self.device)
        self.mean_     = X.mean(0)
        X_c            = X - self.mean_
        _, S, V        = torch.pca_lowrank(X_c, q=self.n_components, niter=self.niter)
        self.components_ = V   # (D, n_components)
        total_var      = (X_c * X_c).sum() / (len(X) - 1)
        self.explained_variance_ratio_ = (S ** 2 / (len(X) - 1)) / total_var
        return self

    @torch.no_grad()
    def transform(self, X_np):
        X = torch.from_numpy(X_np).to(self.device)
        return ((X - self.mean_) @ self.components_).cpu().float().numpy()

# ── DINOv2 extractor ──────────────────────────────────────────────────────────

class DINOv2(nn.Module):
    def __init__(self, backbone_name, layers, grid_size, device='cuda'):
        super().__init__()
        self.backbone_name = backbone_name
        self.layers = layers
        self.grid_size = grid_size
        self.device = device
        
        try:
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        except Exception as e:
            fallback = backbone_name.replace('_reg4', '')
            print(f'Fallback to {fallback}: {e}')
            self.backbone = timm.create_model(fallback, pretrained=True, num_classes=0)
            
        for p in self.backbone.parameters():
            p.requires_grad_(False)
            
        sig            = inspect.signature(self.backbone.get_intermediate_layers)
        self._new_api  = 'return_class_token' in sig.parameters
        self._n_blocks = len(self.backbone.blocks)
        print(f'DINOv2  blocks={self._n_blocks}  new_api={self._new_api}')

    @torch.no_grad()
    def forward(self, x):
        """x: (B, 3, H, W) -> (B, L, FEAT_DIM_RAW) float32"""
        if self._new_api:
            outs   = self.backbone.get_intermediate_layers(
                x, n=self.layers, reshape=False, return_class_token=False, norm=True)
            tokens = [o[:, -self.grid_size * self.grid_size:, :] for o in outs]
        else:
            outs   = self.backbone.get_intermediate_layers(x, n=self._n_blocks)
            tokens = [outs[i][:, -self.grid_size * self.grid_size:, :] for i in self.layers]
        return torch.cat(tokens, dim=-1).float()

# ── Scoring utilities ──────────────────────────────────────────────────────────

def _gauss_kernel(sigma, device='cuda'):
    ks = int(6 * sigma + 1) | 1
    x  = torch.arange(ks, dtype=torch.float32) - ks // 2
    g  = torch.exp(-x ** 2 / (2 * sigma ** 2));  g /= g.sum()
    return g.outer(g).view(1, 1, ks, ks).to(device)

@torch.no_grad()
def score_patches(feat_ND, bank, device='cuda', knn_k=3, score_chunk=2048):
    """(N, FEAT_DIM) numpy -> (N,) numpy — chunked GPU matmul k-NN."""
    q       = torch.from_numpy(feat_ND).to(device)
    bank_sq = (bank * bank).sum(1)   # (M,) precomputed once per call
    out     = []
    for i in range(0, len(q), score_chunk):
        qi = q[i : i + score_chunk]
        d  = ((qi * qi).sum(1, keepdim=True) + bank_sq
              - 2.0 * (qi @ bank.T)).clamp(min=0)   # (chunk, M)
        out.append(torch.topk(d, knn_k, dim=1, largest=False).values.mean(1))
    return torch.cat(out).cpu().numpy()

def make_heatmaps(scores_BN, grid_size, smooth_sigma=0.5, device='cuda', out_size=224):
    """(B, GRID^2) numpy -> (B, OUT, OUT) numpy — batched Gaussian + bilinear upsample."""
    B = scores_BN.shape[0]
    t = torch.from_numpy(
        scores_BN.reshape(B, 1, grid_size, grid_size).astype(np.float32)).to(device)
    
    gauss = _gauss_kernel(smooth_sigma, device=device)
    pad = gauss.shape[-1] // 2
    
    t = F.conv2d(t, gauss, padding=pad)
    return F.interpolate(t, (out_size, out_size), mode='bilinear',
                         align_corners=False).squeeze(1).cpu().numpy()

# ── q8rle encoding ────────────────────────────────────────────────────────────

def rle_encode_q8(arr):
    q      = np.clip(np.rint(arr * 255), 0, 255).astype(np.uint8)
    h, w   = q.shape
    flat   = q.T.reshape(-1)   # column-major (Fortran order)
    cuts   = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends   = np.r_[cuts, flat.size]
    parts  = ['q8rle', str(h), str(w)]
    for val, cnt in zip(flat[starts], ends - starts):
        parts += [str(int(val)), str(int(cnt))]
    return ' '.join(parts)

def load_img(path, img_size=518):
    """Load and preprocess image for DINOv2."""
    transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(path).convert('RGB'))
