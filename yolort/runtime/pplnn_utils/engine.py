# Copyright (c) 2021, The OpenPPL teams.
# Copyright (c) 2021, Zhiqiang Wang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from pyppl import nn as pplnn, common as pplcommon
except ImportError:
    pplnn, pplcommon = None, None


def register_engines(
    use_x86=True,
    disable_avx512=False,
    use_cuda=False,
    device_id=0,
    quick_select=False,
):
    """Register engines"""
    if use_cuda:
        return register_engines_gpu(device_id=device_id, quick_select=quick_select)
    elif use_x86:
        return register_engines_cpu(disable_avx512=disable_avx512)
    else:
        raise NotImplementedError("Currently not supports this device")


def register_engines_cpu(disable_avx512=False):
    engines = []
    x86_engine = pplnn.X86EngineFactory.Create()
    if not x86_engine:
        raise RuntimeError("Create x86 engine failed.")

    if disable_avx512:
        status = x86_engine.Configure(pplnn.X86_CONF_DISABLE_AVX512)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"x86 engine Configure() failed: {pplcommon.GetRetCodeStr(status)}")

    engines.append(pplnn.Engine(x86_engine))

    return engines


def register_engines_gpu(device_id=0, quick_select=False):
    engines = []

    cuda_options = pplnn.CudaEngineOptions()
    cuda_options.device_id = device_id

    cuda_engine = pplnn.CudaEngineFactory.Create(cuda_options)
    if not cuda_engine:
        raise RuntimeError("Create cuda engine failed.")

    if quick_select:
        status = cuda_engine.Configure(pplnn.CUDA_CONF_USE_DEFAULT_ALGORITHMS)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"cuda engine Configure() failed: {pplcommon.GetRetCodeStr(status)}")

    engines.append(pplnn.Engine(cuda_engine))

    return engines
