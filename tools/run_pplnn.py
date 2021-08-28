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

from pathlib import Path
import argparse
import logging

try:
    from pyppl import nn as pplnn, common as pplcommon
except ImportError:
    pplnn, pplcommon = None, None

from yolort.runtime.pplnn_utils.logger import print_input_output_info
from yolort.runtime.pplnn_utils.data import (
    set_input_one_by_one,
    set_random_inputs,
    set_reshaped_inputs_one_by_one,
    save_inputs_all_in_one,
    save_inputs_one_by_one,
    save_outputs_one_by_one,
)
from yolort.runtime.pplnn_utils.engine import register_engines


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", dest="display_version", action="store_true")
    # X86
    parser.add_argument("--use_x86", action="store_true")
    parser.add_argument("--disable_avx512", action="store_true")
    # CUDA
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--quick_select", action="store_true")
    parser.add_argument("--device_id", type=int, default=0,
                        help="Specify the device id to be used.")

    parser.add_argument("--onnx_model", required=True,
                        help="Path of the onnx model file")
    parser.add_argument("--in_shapes", type=str, default=None,
                        help="Shapes of input tensors. dims are separated "
                             "by underline, inputs are separated by comma. "
                             "example: 1_3_128_128, 2_3_400_640, 3_3_768_1024")

    parser.add_argument("--inputs", type=str, default=None,
                        help="The input files are separated by comma.")
    parser.add_argument("--reshaped_inputs", type=str, default=None,
                        help="Binary input files separated by comma. file name "
                             "format: 'name-dims-datatype.dat'. for example: "
                             "input1-1_1_1_1-fp32.dat, input2-1_1_1_1-fp16.dat or "
                             "input3-1_1-int8.dat")

    parser.add_argument("--save_input", action="store_true",
                        help="Switch used to save all input tensors to NDARRAY "
                             "format in one file named 'pplnn_inputs.dat'")
    parser.add_argument("--save_inputs", action="store_true",
                        help="Switch used to save separated input tensors to NDARRAY format")
    parser.add_argument("--save_outputs", action="store_true",
                        help="Switch used to save separated output tensors to NDARRAY format")
    parser.add_argument("--output_path", type=str, default="./",
                        help="The directory to save input/output data if '--save_*' "
                             "options are enabled.")
    return parser


def parse_in_shapes(in_shapes_str):
    shape_strs = in_shapes_str.split(",") if in_shapes_str else []
    ret = []
    for s in shape_strs:
        dims = [int(d) for d in s.split("_")]
        ret.append(dims)
    return ret


def gen_dims_str(dims):
    if not dims:
        return ""

    s = str(dims[0])
    for i in range(1, len(dims)):
        s = s + "_" + str(dims[i])
    return s


def cli_main():

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = get_args_parser()
    args = parser.parse_args()

    if args.display_version:
        logging.info(f"PPLNN version: {pplnn.GetVersionString()}")

    # Register Engines
    use_x86, use_cuda = args.use_x86, args.use_cuda
    if use_x86 == use_cuda:
        raise NotImplementedError("Currently one and only one device should be enabled.")

    engines = register_engines(
        use_x86=use_x86,
        disable_avx512=args.disable_avx512,
        use_cuda=use_cuda,
        device_id=args.device_id,
        quick_select=args.quick_select,
    )

    # Creating a Runtime Builder
    builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(args.onnx_model, engines)
    if not builder:
        raise RuntimeError("Create OnnxRuntimeBuilder failed.")

    # Creating a Runtime Instance
    runtime_options = pplnn.RuntimeOptions()
    runtime = builder.CreateRuntime(runtime_options)
    if not runtime:
        raise RuntimeError("Create Runtime instance failed.")

    # Filling Input Data to Runtime
    in_shapes = parse_in_shapes(args.in_shapes)

    if args.inputs:
        set_input_one_by_one(args.inputs, in_shapes, runtime)
    elif args.reshaped_inputs:
        set_reshaped_inputs_one_by_one(args.reshaped_inputs, runtime)
    else:
        set_random_inputs(in_shapes, runtime)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.save_input:
        save_inputs_all_in_one(output_path, runtime)
    if args.save_inputs:
        save_inputs_one_by_one(output_path, runtime)

    # Forward
    status = runtime.Run()
    if status != pplcommon.RC_SUCCESS:
        raise RuntimeError(f"Run() failed: {pplcommon.GetRetCodeStr(status)}")

    status = runtime.Sync()  # wait for all ops run finished
    if status != pplcommon.RC_SUCCESS:
        raise RuntimeError(f"Run() failed: {pplcommon.GetRetCodeStr(status)}")

    logging.info("Successfully run network!")

    # Getting Results
    print_input_output_info(runtime)

    if args.save_outputs:
        save_outputs_one_by_one(output_path, runtime)


if __name__ == "__main__":
    cli_main()
