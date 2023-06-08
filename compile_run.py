#!/usr/bin/env python3
import logging
import os
from furiosa import runtime
from furiosa.runtime import session
import numpy as np

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

quantized_model_path = './unet_model_quantized.dfg'
def run_example():
    runtime.__full_version__

    sess = session.create(str(quantized_model_path))
    sess.print_summary()
    
    # Print the first input tensor shape and dimensions
    input_tensor = sess.inputs()[0]
    print(f'input tensor: {input_tensor}')
    
    # Generate the random input tensor according to the input shape
    input = np.random.randint(0, 255, input_tensor.shape).astype("float32")
    
    # Run the inference
    outputs = sess.run(input)
    
    print("== Output ==")
    print(outputs)
    print(outputs[0].numpy())


if __name__ == "__main__":
    run_example()
