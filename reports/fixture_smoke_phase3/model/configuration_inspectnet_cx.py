from contextlib import suppress

from transformers import PretrainedConfig


class InspectNetCXConfig(PretrainedConfig):
    model_type = "inspectnet-cx"

    def __init__(
        self,
        backbone: str = "phase0-tiny-cnn",
        image_size: int = 224,
        feature_layers: list[str] | None = None,
        prototype_size: int = 128,
        calibration_method: str = "normal_quantile",
        threshold: float = 0.5,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.backbone = backbone
        self.image_size = image_size
        self.feature_layers = feature_layers or ["stage2", "stage3"]
        self.prototype_size = prototype_size
        self.calibration_method = calibration_method
        self.threshold = threshold


with suppress(AttributeError, ValueError):
    InspectNetCXConfig.register_for_auto_class()
