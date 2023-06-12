#!/bin/bash


if [[ $1 -eq 0 ]]
then
    echo "Usage: $0 <batch-size>"
else
    echo "+++Converting to ONNX"
    ./convert_2_onnx.py --bs $1

    echo "+++Quantizing from ONNX"
    ./quantize.py --bs $1

    echo "+++Compiling and running"
    ./async_compile_run.py --bs $1
fi
