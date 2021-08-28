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

import logging

try:
    from pyppl import common as pplcommon
except ImportError:
    pplcommon = None


def logging_info(prefix, i, tensor, shape, dims):
    logging.info(f"{prefix}[{i}]")
    logging.info(f"\tname: {tensor.GetName()}")
    logging.info(f"\tdim(s): {dims}")
    logging.info(f"\ttype: {pplcommon.GetDataTypeStr(shape.GetDataType())}")
    logging.info(f"\tformat: {pplcommon.GetDataFormatStr(shape.GetDataFormat())}")
    byte_excluding_padding = calc_bytes(dims, pplcommon.GetSizeOfDataType(shape.GetDataType()))
    logging.info(f"\tbyte(s) excluding padding: {byte_excluding_padding}")


def calc_bytes(dims, item_size):
    nbytes = item_size
    for d in dims:
        nbytes = nbytes * d
    return nbytes


def print_input_output_info(runtime):
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        dims = shape.GetDims()
        logging_info("input", i, tensor, shape, dims)

    for i in range(runtime.GetOutputCount()):
        tensor = runtime.GetOutputTensor(i)
        shape = tensor.GetShape()
        dims = shape.GetDims()
        logging_info("output", i, tensor, shape, dims)
