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

import numpy as np

try:
    from pyppl import common as pplcommon
except ImportError:
    pplcommon = None


PPLNN_DATA_TYPE_STR_MAPS = {
    pplcommon.DATATYPE_INT8: "int8",
    pplcommon.DATATYPE_INT16: "int16",
    pplcommon.DATATYPE_INT32: "int32",
    pplcommon.DATATYPE_INT64: "int64",
    pplcommon.DATATYPE_UINT8: "uint8",
    pplcommon.DATATYPE_UINT16: "uint16",
    pplcommon.DATATYPE_UINT32: "uint32",
    pplcommon.DATATYPE_UINT64: "uint64",
    pplcommon.DATATYPE_FLOAT16: "fp16",
    pplcommon.DATATYPE_FLOAT32: "fp32",
    pplcommon.DATATYPE_FLOAT64: "fp64",
    pplcommon.DATATYPE_BOOL: "bool",
    pplcommon.DATATYPE_UNKNOWN: "unknown",
}


PPLNN_DATA_TYPE_NUMPY_MAPS = {
    pplcommon.DATATYPE_INT8: np.int8,
    pplcommon.DATATYPE_INT16: np.int16,
    pplcommon.DATATYPE_INT32: np.int32,
    pplcommon.DATATYPE_INT64: np.int64,
    pplcommon.DATATYPE_UINT8: np.uint8,
    pplcommon.DATATYPE_UINT16: np.uint16,
    pplcommon.DATATYPE_UINT32: np.uint32,
    pplcommon.DATATYPE_UINT64: np.uint64,
    pplcommon.DATATYPE_FLOAT16: np.float16,
    pplcommon.DATATYPE_FLOAT32: np.float32,
    pplcommon.DATATYPE_FLOAT64: np.float64,
    pplcommon.DATATYPE_BOOL: bool,
}
