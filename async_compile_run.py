#!/usr/bin/env python3
import logging
import os
from furiosa import runtime
from furiosa.runtime import session
import numpy as np
import time

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

import argparse

parser = argparse.ArgumentParser(
                    prog='async_compile_run.py',
                    description='compile and run',
                    epilog='look for quantized file with .dfg suffix')

parser.add_argument('--bs', required=True, type=int)
args = parser.parse_args()
print(f'requested batch size: {args.bs}')
batch_size = args.bs

nb_batches = 100

quantized_model_path = './unet_model_quantized.dfg'
def run_async():
    runtime.__full_version__

    submitter, queue = session.create_async(str(quantized_model_path),
                                worker_num=1,
                                input_queue_size=100 * batch_size, # requests you can submit without blocking
                                            output_queue_size=100 * batch_size)
    print(f'input: {submitter.inputs()[0]}')
    print(f'output: {submitter.outputs()[0]}')    
    
    input_tensor = submitter.inputs()[0]
    tic = time.perf_counter()
    # Submit the inference requests asynchronously
    for i in range(0, nb_batches):
        # Generate the random input tensor according to the input shape and submit
        input = np.random.randint(0, 255, input_tensor.shape).astype("float32")
        submitter.submit(input, context=i)
    
    # Receive the results asynchronously
    context_list = []
    output_list = []
    for i in range(0, nb_batches):
        context, outputs = queue.recv(100) # provide timeout param. If None, queue.recv() will be blocking.
        context_list.append(context)
        output_list.append(outputs)
    
    toc = time.perf_counter()
    
    print("== Output ==")
    for context, outputs in zip(context_list, output_list):
        print(f"Context: {context}, Predict: {np.argmax(outputs[0].numpy())}")

    # housekeeping
    if queue:
        queue.close()
    if submitter:
        submitter.close()
    
    print(f"Completed {nb_batches} inference of unet in {toc - tic:0.4f} seconds")
    print(f'Unet inference through put: {batch_size * (1/((toc - tic) / nb_batches)):0.4}/sec')


if __name__ == "__main__":
    run_async()
