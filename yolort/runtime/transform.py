# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from abc import ABCMeta, abstractmethod


class TransformBase(metaclass=ABCMeta):
    """
    YOLO Runtime Transform Wrapper.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def resize(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass
