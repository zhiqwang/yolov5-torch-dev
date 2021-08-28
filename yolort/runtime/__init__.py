# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.

from .utils import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole runtime.

The registered object will be called with `obj(cfg)`
and expected to return a `RuntimeBase` object.
"""
