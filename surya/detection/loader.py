from typing import Optional

import torch

from surya.common.load import ModelLoader
from surya.detection.processor import SegformerImageProcessor

from surya.detection.model.config import EfficientViTConfig
from surya.detection.model.encoderdecoder import EfficientViTForSemanticSegmentation
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


class DetectionModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.DETECTOR_MODEL_CHECKPOINT

    def model(
        self,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype | str] = None,
        attention_implementation: Optional[str] = None,
    ) -> EfficientViTForSemanticSegmentation:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        config = EfficientViTConfig.from_pretrained(self.checkpoint)
        model = EfficientViTForSemanticSegmentation.from_pretrained(
            self.checkpoint,
            dtype=dtype,
            config=config,
        )

        # Handle meta device properly
        try:
            # Use next() with iterator to avoid materializing all params
            first_param = next(iter(model.parameters()))
            if first_param.device.type == "meta":
                model = model.to_empty(device=device)
            else:
                model = model.to(device)
        except StopIteration:
            # Model has no parameters, safe to use regular .to()
            model = model.to(device)

        model = model.eval()

        if settings.COMPILE_ALL or settings.COMPILE_DETECTOR:
            torch._dynamo.config.cache_size_limit = 1
            torch._dynamo.config.suppress_errors = False

            logger.info(
                f"Compiling detection model {self.checkpoint} on device {device} with dtype {dtype}"
            )
            compile_args = {"backend": "openxla"} if device == "xla" else {}
            model = torch.compile(model, **compile_args)

        logger.debug(
            f"Loaded detection model {self.checkpoint} from {EfficientViTForSemanticSegmentation.get_local_path(self.checkpoint)} onto device {device} with dtype {dtype}"
        )
        return model

    def processor(
        self,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype | str] = None,
    ) -> SegformerImageProcessor:
        return SegformerImageProcessor.from_pretrained(self.checkpoint)
