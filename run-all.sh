#!/bin/bash


if [[ $1 -eq 0 ]]
then
    echo "Usage: $0 <batch-size>"
else
    ./convert_2_onnx.py --bs $1

    ./quantize.py --bs $1

    ./async_compile_run.py --bs $1
fi
