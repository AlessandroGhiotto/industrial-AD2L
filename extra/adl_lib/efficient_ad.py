from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class PDN(nn.Module):
    def __init__(self, out_channels=384, padding=False):
        super().__init__()
        p = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=p)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=p)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=p)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avg_pool(x)
        x = F.relu(self.conv2(x))
        x = self.avg_pool(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, out_channels=384):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=1), nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=4, stride=1), nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


DEFAULT_TEACHER_CKPT = "./checkpoints/efficient_ad/teacher_small.pth"


class EfficientAD(nn.Module):
    def __init__(
        self,
        out_channels=384,
        teacher_ckpt: str | None = DEFAULT_TEACHER_CKPT,
        imagenet_normalize: bool = True,
    ):
        super().__init__()
        self.teacher = PDN(out_channels=out_channels)
        self.student = PDN(out_channels=out_channels)
        self.autoencoder = AutoEncoder(out_channels=out_channels)
        self.imagenet_normalize = imagenet_normalize

        # Load pre-distilled teacher weights. Without this, the teacher emits noise
        # and the ST anomaly stream carries no signal.
        if teacher_ckpt is None:
            raise ValueError(
                "EfficientAD requires a pre-distilled teacher checkpoint. "
                "Pass teacher_ckpt=<path>. Anomalib's teacher_small.pth works as "
                "a drop-in for out_channels=384."
            )
        ckpt_path = Path(teacher_ckpt)
        if not ckpt_path.exists():
            print(f"Teacher checkpoint not found at {ckpt_path}. Downloading...")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            import zipfile
            import tempfile
            url = "https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = Path(tmp_dir) / "weights.zip"
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    with zip_ref.open("efficientad_pretrained_weights/pretrained_teacher_small.pth") as source, open(ckpt_path, "wb") as target:
                        target.write(source.read())
            print(f"Successfully downloaded teacher checkpoint to {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.teacher.load_state_dict(state, strict=True)

        # Teacher is always frozen
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # ImageNet normalization buffers
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Running mean/std for teacher feature normalization (channel-wise).
        self.register_buffer("teacher_mean", torch.zeros(1, out_channels, 1, 1))
        self.register_buffer("teacher_std", torch.ones(1, out_channels, 1, 1))
        self.register_buffer("teacher_stats_initialized", torch.tensor(False))

        # Independent per-stream quantile buffers (paper's q_a, q_b). Filled by
        # compute_map_quantiles() after training, used at inference to normalize
        # the ST and AE maps to comparable scales before summing.
        self.register_buffer("st_q_a", torch.tensor(0.0))
        self.register_buffer("st_q_b", torch.tensor(1.0))
        self.register_buffer("ae_q_a", torch.tensor(0.0))
        self.register_buffer("ae_q_b", torch.tensor(1.0))
        self.register_buffer("map_quantiles_initialized", torch.tensor(False))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _preprocess(self, x):
        if self.imagenet_normalize:
            x = (x - self.img_mean) / self.img_std
        return x

    def _teacher_features(self, x, update_stats: bool):
        with torch.no_grad():
            t = self.teacher(self._preprocess(x))
            if update_stats:
                batch_mean = torch.mean(t, dim=(0, 2, 3), keepdim=True)
                batch_std = torch.std(t, dim=(0, 2, 3), keepdim=True)
                if not bool(self.teacher_stats_initialized):
                    self.teacher_mean.copy_(batch_mean)
                    self.teacher_std.copy_(batch_std)
                    self.teacher_stats_initialized.fill_(True)
                else:
                    momentum = 0.1
                    self.teacher_mean.mul_(1 - momentum).add_(batch_mean * momentum)
                    self.teacher_std.mul_(1 - momentum).add_(batch_std * momentum)
            t = (t - self.teacher_mean) / (self.teacher_std + 1e-5)
        return t

    def forward(self, x):
        teacher_features = self._teacher_features(x, update_stats=self.training)
        student_features = self.student(self._preprocess(x))
        ae_features = self.autoencoder(self._preprocess(x))

        if self.training:
            return teacher_features, student_features, ae_features

        st_diff = torch.mean((teacher_features - student_features) ** 2, dim=1, keepdim=True)
        ae_diff = torch.mean((teacher_features - ae_features) ** 2, dim=1, keepdim=True)

        if bool(self.map_quantiles_initialized):
            # Paper's normalization: scale each stream so that q_a maps to 0 and
            # q_b maps to 0.1, then sum. Keeps the two streams on comparable scales.
            st_norm = 0.1 * (st_diff - self.st_q_a) / (self.st_q_b - self.st_q_a + 1e-6)
            ae_norm = 0.1 * (ae_diff - self.ae_q_a) / (self.ae_q_b - self.ae_q_a + 1e-6)
            combined = st_norm + ae_norm
        else:
            combined = st_diff + ae_diff

        anomaly_map = F.interpolate(
            combined, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return anomaly_map

    @torch.no_grad()
    def compute_stream_maps(self, x):
        """Return (st_diff, ae_diff) at feature resolution, without combining.

        Used for computing per-stream quantiles on a normal validation set
        and for the supervised auxiliary loss path.
        """
        teacher_features = self._teacher_features(x, update_stats=False)
        student_features = self.student(self._preprocess(x))
        ae_features = self.autoencoder(self._preprocess(x))
        st_diff = torch.mean((teacher_features - student_features) ** 2, dim=1, keepdim=True)
        ae_diff = torch.mean((teacher_features - ae_features) ** 2, dim=1, keepdim=True)
        return st_diff, ae_diff

    @torch.no_grad()
    def compute_map_quantiles(self, loader, device, q_a=0.9, q_b=0.995):
        """Calibrate per-stream quantiles on a normal-image loader."""
        self.eval()
        st_vals, ae_vals = [], []
        for batch in loader:
            images = batch["image"].to(device)
            st, ae = self.compute_stream_maps(images)
            st_vals.append(st.flatten().cpu())
            ae_vals.append(ae.flatten().cpu())
        st_all = torch.cat(st_vals)
        ae_all = torch.cat(ae_vals)
        self.st_q_a.fill_(float(torch.quantile(st_all, q_a)))
        self.st_q_b.fill_(float(torch.quantile(st_all, q_b)))
        self.ae_q_a.fill_(float(torch.quantile(ae_all, q_a)))
        self.ae_q_b.fill_(float(torch.quantile(ae_all, q_b)))
        self.map_quantiles_initialized.fill_(True)
