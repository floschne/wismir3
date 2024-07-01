from PIL import Image

from .image_transformation_base import ImageTransformationBase


# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#webp
class WebPTransformation(ImageTransformationBase):
    def __init__(self, lossless: bool = True, quality: int = 50, method=6):
        super().__init__("WebP")
        self.lossless = lossless
        self.quality = quality
        self.method = method

    def apply(self, img: Image, **kwargs) -> Image:
        path = kwargs['img_path']
        if '.png' in path:
            path = path.replace('.png', '.webp')
        assert path[-5:] == '.webp', f"Cannot set webp extension to image path {path}"

        img.save(path,
                 lossless=self.lossless,
                 quality=self.quality,
                 method=self.method)
        return img
