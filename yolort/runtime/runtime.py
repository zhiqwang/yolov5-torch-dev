# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import pplnn
from .pplnn_utils.engine import register_engines_cpu, register_engines_gpu
from .pplnn_utils.data import (
    set_input_one_by_one,
    set_random_inputs,
    set_reshaped_inputs_one_by_one,
    save_inputs_all_in_one,
    save_inputs_one_by_one,
    save_outputs_one_by_one,
)


class RuntimeBase:
    """
    YOLO Runtime Wrapper.
    """
    def __init__(self, engine=None, transform=None):
        self._engine = engine
        self._transform = transform

    @property
    def engine(self):
        return self._engine

    @property
    def transform(self):
        return self._transform


class RuntimePPLNN(RuntimeBase):
    def __init__(
        self,
        use_x86=True,
        disable_avx512=False,
        use_cuda=False,
        device_id=0,
        quick_select=False,
    ):
        """Register engines"""
        if use_cuda:
            engines = register_engines_gpu(device_id=device_id, quick_select=quick_select)
        elif use_x86:
            engines = register_engines_cpu(disable_avx512=disable_avx512)
        else:
            raise NotImplementedError("Currently not supports this device")
        builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(args.onnx_model, engines)

        super().__init__(engines)
