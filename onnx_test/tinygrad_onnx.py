#!/usr/bin/env python3
import os
#from ultralytics import YOLO
from pathlib import Path
from tinygrad.nn.onnx import OnnxRunner
from onnx_helpers import get_example_inputs
import time
from tinygrad import TinyJit

run_onnx = OnnxRunner("weights/yolov9-c.onnx")
@TinyJit
def run_onnx_jit(x): return run_onnx(x, debug=True)

if __name__ == "__main__":

    ts = time.perf_counter()
    for _ in range(100):
        run_onnx_jit(get_example_inputs(run_onnx.graph_inputs))

    print(100 / (time.perf_counter() - ts))