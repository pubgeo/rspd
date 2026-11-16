"""Optional paper baselines. All adapters return distances (lower is better)."""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@lru_cache(maxsize=8)
def read_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def create_msssim_scorer():
    """The fixed five-scale implementation used in the paper, not a fallback."""
    from skimage.metrics import structural_similarity

    weights = np.asarray([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)

    def downsample(image):
        height, width = image.shape[:2]
        resized = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).resize(
            (max(1, width // 2), max(1, height // 2)), Image.Resampling.BOX
        )
        return np.asarray(resized, dtype=np.float32)

    def score(reference_path, candidate_path):
        reference = np.asarray(read_rgb(reference_path), dtype=np.float32)
        candidate = np.asarray(read_rgb(candidate_path), dtype=np.float32)
        if reference.shape != candidate.shape:
            raise ValueError("MS-SSIM requires reference and candidate images with matching dimensions.")
        values = []
        for index in range(len(weights)):
            min_dim = min(reference.shape[:2] + candidate.shape[:2])
            if min_dim < 3:
                value = 1.0 if np.array_equal(reference, candidate) else 0.0
            else:
                kwargs = {"channel_axis": 2, "data_range": 255.0}
                if min_dim < 7:
                    kwargs["win_size"] = min_dim if min_dim % 2 else min_dim - 1
                value = float(structural_similarity(reference, candidate, **kwargs))
            values.append(value)
            if index == len(weights) - 1 or min_dim < 2:
                break
            reference, candidate = downsample(reference), downsample(candidate)
        used = weights[:len(values)]
        used = used / used.sum()
        similarity = np.prod(np.power(np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0), used))
        return 1.0 - float(similarity)

    return score


def create_scorers(names: list[str], device: str | None = None, diffsim_root: Path | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorers = {}
    if {"rspd", "rspd_q95"}.intersection(names):
        from rspd import RSPDMean7

        ecdf = RSPDMean7(device=device)
        q95 = RSPDMean7(ecdf.score_model, ecdf.calibration, method="pooled_q95")

        @lru_cache(maxsize=4)
        def details(reference, candidate):
            return ecdf.score_pair(read_rgb(reference), read_rgb(candidate), return_details=True)

        scorers["rspd"] = lambda reference, candidate: details(reference, candidate).score
        scorers["rspd_q95"] = lambda reference, candidate: q95.score_raw_distances(
            details(reference, candidate).raw_layer_distances
        ).score
    if "msssim" in names:
        scorers["msssim"] = create_msssim_scorer()
    if "lpips" in names:
        import lpips
        from torchvision.transforms.functional import to_tensor

        lpips_model = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()

        def score_lpips(reference, candidate):
            with torch.inference_mode():
                a = to_tensor(read_rgb(reference)).unsqueeze(0).to(device).mul(2).sub(1)
                b = to_tensor(read_rgb(candidate)).unsqueeze(0).to(device).mul(2).sub(1)
                return float(lpips_model(a, b).item())

        scorers["lpips"] = score_lpips
    if "dreamsim" in names:
        from dreamsim import dreamsim

        cache = Path(torch.hub.get_dir()) / "dreamsim"
        dreamsim_model, preprocess = dreamsim(
            pretrained=True, device=device, dreamsim_type="ensemble", cache_dir=str(cache)
        )
        dreamsim_model.eval()

        def score_dreamsim(reference, candidate):
            with torch.inference_mode():
                a, b = preprocess(read_rgb(reference)).to(device), preprocess(read_rgb(candidate)).to(device)
                return float(dreamsim_model(a, b).item())

        scorers["dreamsim"] = score_dreamsim
    if "remoteclip" in names:
        from rspd.remoteclip import RemoteCLIPScore

        remoteclip = ecdf.score_model if "rspd" in scorers else RemoteCLIPScore(device=device)

        def score_remoteclip(reference, candidate):
            a = remoteclip.get_image_features(read_rgb(reference), norm=True)
            b = remoteclip.get_image_features(read_rgb(candidate), norm=True)
            return 1.0 - float(torch.nn.functional.cosine_similarity(a, b).item())

        scorers["remoteclip"] = score_remoteclip
    if "diffsim" in names:
        if diffsim_root is None or not (diffsim_root / "diffsim" / "diffsim.py").is_file():
            raise ValueError("DiffSim requires --diffsim-root pointing to its upstream checkout. See docs/EVALUATION.md.")
        sys.path.insert(0, str(diffsim_root.resolve()))
        try:
            from diffsim.diffsim import DiffSim
        finally:
            sys.path.pop(0)

        class DeterministicDiffSim(DiffSim):
            def prepare_image_latents(self, image, pipe, device, generator=None):
                image = image.to(device=device, dtype=self.torch_dtype)
                return pipe.vae.config.scaling_factor * pipe.vae.encode(image).latent_dist.mode()

        model = DeterministicDiffSim(
            device=device, torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
            ip_adapter=False, model_id="runwayml/stable-diffusion-v1-5",
        )

        def score_diffsim(reference, candidate):
            with torch.inference_mode():
                score = model.diffsim(
                    image_A=reference, image_B=candidate, img_size=512, prompt="High quality image",
                    target_block="up_blocks", target_layer=0, target_step=600,
                    ip_adapter=False, seed=2334, device=device, similarity="cosine",
                )
            return 1.0 - float(score.item() if hasattr(score, "item") else score)

        scorers["diffsim"] = score_diffsim
    return {name: scorers[name] for name in names}
