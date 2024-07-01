from PIL import Image

from .image_transformation_base import ImageTransformationBase


class ResizeTransformation(ImageTransformationBase):
    def apply(self, img: Image, **kwargs) -> Image:
        img.thumbnail([self.maxWidth, self.maxHeight], resample=self.resampling)
        return img

    def __init__(self, maxWidth: int, maxHeight: int, resampling: int = Image.BICUBIC):
        super().__init__("Resize")
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        self.resampling = resampling
