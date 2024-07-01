from typing import Tuple

from PIL import Image

from .image_transformation_base import ImageTransformationBase


class CompressionTransformation(ImageTransformationBase):
    def __init__(self, optimize: bool, dpi: Tuple[int, int]):
        super().__init__("Compress")
        self.optimize = optimize
        self.dpi = dpi

    def apply(self, img: Image, **kwargs) -> Image:
        img.save(kwargs['img_path'], optimize=self.optimize, dpi=self.dpi)
        return img
