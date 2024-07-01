from abc import ABC, abstractmethod

from PIL import Image


class ImageTransformationBase(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, img: Image, **kwargs) -> Image:
        pass

    def __call__(self, img: Image, **kwargs) -> Image:
        return self.apply(img, **kwargs)
