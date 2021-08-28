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

from pathlib import PurePath
import random
import numpy as np

try:
    from pyppl import common as pplcommon
except ImportError:
    pplcommon = None

from .builtin_meta import PPLNN_DATA_TYPE_NUMPY_MAPS, PPLNN_DATA_TYPE_STR_MAPS


def set_input_one_by_one(inputs, in_shapes, runtime):
    input_files = inputs.split(",") if inputs else []
    file_num = len(input_files)
    if file_num != runtime.GetInputCount():
        raise RuntimeError(
            f"Input file num[{str(file_num)}] != graph input num[{runtime.GetInputCount()}]")

    for i in range(file_num):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        np_data_type = PPLNN_DATA_TYPE_NUMPY_MAPS[shape.GetDataType()]

        dims = []
        if in_shapes:
            dims = in_shapes[i]
        else:
            dims = shape.GetDims()

        in_data = np.fromfile(input_files[i], dtype=np_data_type).reshape(dims)
        # convert data type & format from `in_data` to `tensor` & fill data
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"Copy data to tensor[{tensor.GetName()}] failed: {pplcommon.GetRetCodeStr(status)}")


def set_reshaped_inputs_one_by_one(reshaped_inputs, runtime):
    input_files = reshaped_inputs.split(",") if reshaped_inputs else []
    file_num = len(input_files)
    if file_num != runtime.GetInputCount():
        raise RuntimeError(
            f"Input file num[{str(file_num)}] != graph input num[{runtime.GetInputCount()}]")

    for i in range(file_num):
        input_file_name = PurePath(input_files[i]).name
        file_name_components = input_file_name.split("-")
        if len(file_name_components) != 3:
            raise ValueError(
                f"invalid input filename[{input_files[i]}] in '--reshaped_inputs'.")

        input_shape_str_list = file_name_components[1].split("_")
        input_shape = [int(s) for s in input_shape_str_list]

        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        np_data_type = PPLNN_DATA_TYPE_NUMPY_MAPS[shape.GetDataType()]
        in_data = np.fromfile(input_files[i], dtype=np_data_type).reshape(input_shape)
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"Copy data to tensor[{tensor.GetName()}] failed: {pplcommon.GetRetCodeStr(status)}")


def set_random_inputs(in_shapes, runtime):
    def generate_random_dims(shape):
        dims = shape.GetDims()
        dim_count = len(dims)
        for i in range(2, dim_count):
            if dims[i] == 1:
                dims[i] = random.randint(128, 641)
                if dims[i] % 2 != 0:
                    dims[i] = dims[i] + 1
        return dims

    rng = np.random.default_rng()
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        data_type = shape.GetDataType()

        np_data_type = PPLNN_DATA_TYPE_NUMPY_MAPS[data_type]
        if np_data_type in [np.float16, np.float32, np.float64]:
            lower_bound = -1.0
            upper_bound = 1.0
        else:
            info = np.iinfo(np_data_type)
            lower_bound = info.min
            upper_bound = info.max

        dims = []
        if in_shapes:
            dims = in_shapes[i]
        else:
            dims = generate_random_dims(shape)

        in_data = (upper_bound - lower_bound) * rng.random(dims, dtype=np_data_type) * lower_bound
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            raise RuntimeError(
                f"Copy data to tensor[{tensor.GetName()}] failed: {pplcommon.GetRetCodeStr(status)}")


def save_inputs_one_by_one(output_path, runtime):
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        tensor_data = tensor.ConvertToHost()
        if not tensor_data:
            raise RuntimeError(f"Copy data from tensor[{tensor.GetName()}] failed.")

        in_data = np.array(tensor_data, copy=False)
        out_data_path_name = (f"pplnn_input_{i}_{tensor.GetName()}-"
                              f"{gen_dims_str(shape.GetDims())}-"
                              f"{PPLNN_DATA_TYPE_STR_MAPS[shape.GetDataType()]}.dat")
        in_data.tofile(output_path / out_data_path_name)


def save_inputs_all_in_one(output_path, runtime):
    out_file_name = output_path / "pplnn_inputs.dat"
    with open(out_file_name, mode="wb+") as fd:
        for i in range(runtime.GetInputCount()):
            tensor = runtime.GetInputTensor(i)
            tensor_data = tensor.ConvertToHost()
            if not tensor_data:
                raise RuntimeError(f"Copy data from tensor[{tensor.GetName()}] failed.")

            in_data = np.array(tensor_data, copy=False)
            fd.write(in_data.tobytes())


def save_outputs_one_by_one(output_path, runtime):
    for i in range(runtime.GetOutputCount()):
        tensor = runtime.GetOutputTensor(i)
        tensor_data = tensor.ConvertToHost()
        if not tensor_data:
            raise RuntimeError(f"Copy data from tensor[{tensor.GetName()}] failed.")

        out_data = np.array(tensor_data, copy=False)
        out_data.tofile(output_path / f"pplnn_output-{tensor.GetName()}.dat")
