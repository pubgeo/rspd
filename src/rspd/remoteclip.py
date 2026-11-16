from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import hashlib
import logging
import numpy as np
import open_clip
import PIL
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


HF_REPO_ID = "chendelong/RemoteCLIP"
HF_REVISION = "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
CHECKPOINT_SHA256 = {
    "ViT-B-32": "60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4",
    "ViT-L-14": "fcc2a7e21e171f4ffcb7a9c0206b8b74ac0c9eb83c67b576958b7a4ed6c8cecb",
}
MODEL_CHOICES = ("ViT-B-32", "ViT-L-14")
ImageType: TypeAlias = PIL.Image.Image | np.ndarray | torch.Tensor


def resolve_checkpoint(
    model_name: str,
    checkpoint_path: Path | None,
    cache_dir: Path | None = None,
) -> Path:
    filename = f"RemoteCLIP-{model_name}.pt"

    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            revision=HF_REVISION,
            cache_dir=str(cache_dir.expanduser().resolve()) if cache_dir is not None else None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load {filename} from the Hugging Face cache or download it. "
            "Run `python -m rspd.download` with network access first; use HF_HOME to choose the cache location."
        ) from exc
    with Path(downloaded).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256[model_name]:
        raise ValueError(f"RemoteCLIP checkpoint checksum mismatch: {downloaded}. Remove it and download again.")
    return Path(downloaded)


def unwrap_checkpoint(raw_checkpoint: object) -> dict[str, torch.Tensor]:
    checkpoint = raw_checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a state dict.")

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        cleaned[new_key] = value

    if not cleaned:
        raise ValueError("Checkpoint did not contain tensor weights.")

    return cleaned


def to_pil_image(image: ImageType) -> PIL.Image.Image:
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return PIL.Image.fromarray(array).convert("RGB")

    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError("Tensor image batch must have size 1.")
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError("Tensor image must have shape [C,H,W] or [H,W,C].")
        if tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.dtype != np.uint8:
            array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
        return PIL.Image.fromarray(array).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)!r}")


class RemoteCLIPScore:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        checkpoint_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        torch_dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
    ):
        if model_name not in MODEL_CHOICES:
            raise ValueError(
                f"Unsupported RemoteCLIP backbone '{model_name}'. "
                f"Supported models: {', '.join(MODEL_CHOICES)}."
            )

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.device = device
        self.dtype = torch_dtype

        checkpoint = resolve_checkpoint(
            model_name=model_name,
            checkpoint_path=Path(checkpoint_path) if checkpoint_path is not None else None,
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
        )

        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.ERROR)
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=None,
            )
        finally:
            root_logger.setLevel(previous_level)
        state_dict = unwrap_checkpoint(torch.load(checkpoint, map_location="cpu"))
        load_result = model.load_state_dict(state_dict, strict=True)
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if missing or unexpected:
            raise RuntimeError(
                "RemoteCLIP checkpoint load was incomplete. "
                f"Missing keys: {len(missing)}. Unexpected keys: {len(unexpected)}."
            )

        self.model = model.to(device=device, dtype=torch_dtype)
        self.model.eval()
        self.preprocess = preprocess

        visual = getattr(self.model, "visual", None)
        transformer = getattr(visual, "transformer", None)
        resblocks = getattr(transformer, "resblocks", None)
        if resblocks is None:
            raise TypeError("RemoteCLIP visual encoder does not expose ViT transformer blocks.")
        self.resblocks = resblocks

        logging.getLogger(__name__).info("Loaded RemoteCLIP weights from %s for %s.", checkpoint, model_name)

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None):
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
        if dtype is not None:
            self.dtype = dtype
            self.model = self.model.to(dtype=dtype)

    def _preprocess_images(self, images: ImageType | list[ImageType]) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]
        tensors = [self.preprocess(to_pil_image(image)) for image in images]
        return torch.stack(tensors).to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        inputs = self._preprocess_images(image)
        image_features = self.model.encode_image(inputs)
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    @staticmethod
    def _normalize_target_layer(target_layer: int | list[int] | tuple[int, ...]) -> int:
        if isinstance(target_layer, (list, tuple)):
            if len(target_layer) != 1:
                raise ValueError("RemoteCLIP clip_cross expects exactly one target layer.")
            target_layer = target_layer[0]
        return int(target_layer)

    def _get_hidden_states(self, images: ImageType | list[ImageType], target_layer: int) -> tuple[torch.Tensor, torch.nn.Module]:
        target_layer = self._normalize_target_layer(target_layer)
        if target_layer < 0:
            target_layer = len(self.resblocks) + target_layer
        if target_layer < 0 or target_layer >= len(self.resblocks):
            raise IndexError(f"target_layer={target_layer} is out of range for {len(self.resblocks)} blocks.")

        block = self.resblocks[target_layer]
        captured: dict[str, torch.Tensor] = {}

        def hook(module, inputs):
            captured["hidden_states"] = inputs[0].detach()

        handle = block.register_forward_pre_hook(hook)
        try:
            _ = self.get_image_features(images, norm=True)
        finally:
            handle.remove()

        hidden_states = captured.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError("Failed to capture RemoteCLIP transformer hidden states.")

        hidden_states = hidden_states.permute(1, 0, 2).contiguous()
        return hidden_states, block.attn

    @torch.no_grad()
    def get_hidden_states_for_layers(
        self,
        images: ImageType | list[ImageType],
        layers: list[int] | tuple[int, ...],
    ) -> dict[int, tuple[torch.Tensor, torch.nn.Module]]:
        normalized_layers = list(
            dict.fromkeys(
                len(self.resblocks) + int(layer) if int(layer) < 0 else int(layer)
                for layer in layers
            )
        )
        if not normalized_layers:
            raise ValueError("At least one target layer is required.")
        invalid_layers = [layer for layer in normalized_layers if layer < 0 or layer >= len(self.resblocks)]
        if invalid_layers:
            raise IndexError(f"Target layers are out of range for {len(self.resblocks)} blocks: {invalid_layers}")

        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer_index in normalized_layers:
            block = self.resblocks[layer_index]

            def hook(module, inputs, *, captured_layer: int = layer_index):
                captured[captured_layer] = inputs[0].detach().permute(1, 0, 2).contiguous()

            handles.append(block.register_forward_pre_hook(hook))

        try:
            _ = self.get_image_features(images, norm=True)
        finally:
            for handle in handles:
                handle.remove()

        missing_layers = [layer for layer in normalized_layers if layer not in captured]
        if missing_layers:
            raise RuntimeError(f"Failed to capture RemoteCLIP hidden states for layers: {missing_layers}")
        return {
            layer: (captured[layer], self.resblocks[layer].attn)
            for layer in normalized_layers
        }

    @staticmethod
    def _project_qkv(
        hidden_states: torch.Tensor,
        attention_module: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        batch_size, sequence_length, embed_dim = hidden_states.size()
        qkv = F.linear(
            hidden_states,
            attention_module.in_proj_weight,
            attention_module.in_proj_bias,
        )
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        num_heads = attention_module.num_heads
        head_dim = embed_dim // num_heads
        query_states = query_states.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)

        return query_states, key_states, value_states, (batch_size, sequence_length, embed_dim)

    @staticmethod
    def attention_calc(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_size_shape: tuple[int, int, int],
        out_proj: torch.nn.Module,
    ) -> torch.Tensor:
        batch_size, sequence_length, embed_dim = hidden_size_shape
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, sequence_length, embed_dim)
        attn_output = out_proj(attn_output)
        return attn_output

    def score_from_hidden_states(
        self,
        hidden_a: torch.Tensor,
        hidden_b: torch.Tensor,
        attention_module: torch.nn.Module,
    ) -> torch.Tensor:
        if hidden_a.shape != hidden_b.shape:
            raise ValueError(
                f"Hidden-state batches must have the same shape, got {tuple(hidden_a.shape)} "
                f"and {tuple(hidden_b.shape)}."
            )

        def score_single(single_a: torch.Tensor, single_b: torch.Tensor) -> torch.Tensor:
            query_a, key_a, value_a, hidden_shape = self._project_qkv(single_a, attention_module)
            query_b, key_b, value_b, _ = self._project_qkv(single_b, attention_module)

            attn_a_on_b = self.attention_calc(query_a, key_b, value_b, hidden_shape, attention_module.out_proj)
            attn_b_on_a = self.attention_calc(query_b, key_a, value_a, hidden_shape, attention_module.out_proj)
            self_attn_a = self.attention_calc(query_a, key_a, value_a, hidden_shape, attention_module.out_proj)
            self_attn_b = self.attention_calc(query_b, key_b, value_b, hidden_shape, attention_module.out_proj)

            score_a = F.cosine_similarity(attn_a_on_b.flatten().unsqueeze(0), self_attn_a.flatten().unsqueeze(0))
            score_b = F.cosine_similarity(attn_b_on_a.flatten().unsqueeze(0), self_attn_b.flatten().unsqueeze(0))
            return (score_a + score_b) / 2

        # The original scorer captures sequence-first states after its
        # unconditional permutation. Keep that layout and singleton math
        # unchanged while returning one score per paired image.
        batch_size = hidden_a.shape[1]
        if batch_size == 1:
            return score_single(hidden_a, hidden_b)
        return torch.cat(
            [
                score_single(hidden_a[:, index : index + 1, :], hidden_b[:, index : index + 1, :])
                for index in range(batch_size)
            ]
        )

    @torch.no_grad()
    def clip_cross_scores(
        self,
        images1: ImageType | list[ImageType],
        images2: ImageType | list[ImageType],
        target_layers: list[int] | tuple[int, ...],
    ) -> dict[int, torch.Tensor]:
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        if len(images1) != len(images2):
            raise ValueError(
                f"Number of images1 ({len(images1)}) and images2 ({len(images2)}) should be the same."
            )

        layers = list(dict.fromkeys(int(layer) for layer in target_layers))
        hidden_a = self.get_hidden_states_for_layers(images1, layers)
        hidden_b = self.get_hidden_states_for_layers(images2, layers)
        return {
            layer: self.score_from_hidden_states(
                hidden_a[layer][0],
                hidden_b[layer][0],
                hidden_a[layer][1],
            )
            for layer in layers
        }

    @torch.no_grad()
    def clip_cross_score(
        self,
        images1: ImageType | list[ImageType],
        images2: ImageType | list[ImageType],
        target_layer: int | list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        target_layer = self._normalize_target_layer(target_layer)
        if target_layer < 0:
            target_layer = len(self.resblocks) + target_layer
        if target_layer < 0 or target_layer >= len(self.resblocks):
            raise IndexError(f"target_layer is out of range for {len(self.resblocks)} blocks: {target_layer}")
        return self.clip_cross_scores(images1, images2, [target_layer])[target_layer]
