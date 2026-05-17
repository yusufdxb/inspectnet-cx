import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class InspectNetCXProcessor:
    config_name = "preprocessor_config.json"

    def __init__(
        self,
        image_size: int = 224,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std

    def __call__(
        self,
        images: Any,
        support_images: Any | None = None,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            msg = "Only return_tensors='pt' is supported in Phase 0."
            raise ValueError(msg)

        batch = self._batch(images)
        output = {"pixel_values": batch}
        if support_images is not None:
            output["support_pixel_values"] = self._batch(support_images)
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "image_size": self.image_size,
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "processor_class": self.__class__.__name__,
        }
        (save_path / self.config_name).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *_: object,
        **__: object,
    ) -> "InspectNetCXProcessor":
        config_path = Path(pretrained_model_name_or_path) / cls.config_name
        payload = json.loads(config_path.read_text())
        return cls(
            image_size=int(payload.get("image_size", 224)),
            image_mean=tuple(payload.get("image_mean", (0.485, 0.456, 0.406))),
            image_std=tuple(payload.get("image_std", (0.229, 0.224, 0.225))),
        )

    def _batch(self, images: Any) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            tensor = images
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4:
                msg = "Tensor images must have shape CxHxW or NxCxHxW."
                raise ValueError(msg)
            return tensor.float()

        image_list = [images] if isinstance(images, (Image.Image, np.ndarray)) else list(images)

        tensors = [self._preprocess_one(image) for image in image_list]
        return torch.stack(tensors, dim=0)

    def _preprocess_one(self, image: Image.Image | np.ndarray) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        if not isinstance(image, Image.Image):
            msg = "Images must be PIL images, NumPy arrays, tensors, or lists of those."
            raise TypeError(msg)

        image = image.convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(self.image_mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.image_std, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std
