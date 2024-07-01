from abc import ABC, abstractmethod


class TextTransformationBase(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, txt: str, **kwargs) -> str:
        pass

    def __call__(self, txt: str, **kwargs) -> str:
        return self.apply(txt, **kwargs)
