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
print(f'Requested batch size: {args.bs}')

nb_batches = 600

quantized_model_path = './unet_model_quantized.dfg'
def run_async():
    runtime.__full_version__

    submitter, queue = session.create_async(str(quantized_model_path),
                                            device='npu0pe0',
                                            worker_num=12,
                                            input_queue_size=nb_batches * args.bs, # requests you can submit without blocking
                                            output_queue_size=nb_batches * args.bs,
                                            compiler_config={'graph_lowering_scheme': 'LowerAfterSplit'})
    print(f'input: {submitter.inputs()[0]}')
    print(f'output: {submitter.outputs()[0]}')    
    
    input_tensor = submitter.inputs()[0]

    # Submit the inference requests asynchronously
    inputs = []
    for i in range(0, nb_batches):
        # Generate the random input tensor according to the input shape and submit
        inputs.append(np.random.randint(0, 255, input_tensor.shape).astype("float32"))

    tic = time.perf_counter()
    for i in range(0, nb_batches):
        submitter.submit(inputs[i], context=i)
        
    # Receive the results asynchronously
    context_list = []
    output_list = []
    for i in range(0, nb_batches):
        context, outputs = queue.recv(400) # provide timeout param. If None, queue.recv() will be blocking.
        context_list.append(context)
        output_list.append(outputs.numpy())  # https://github.com/furiosa-ai/furiosa-sdk-private/issues/439)
    
    toc = time.perf_counter()
    
    print("== Output ==")
    for context, outputs in zip(context_list, output_list):
        print(f"Context: {context}, Predict: {np.argmax(outputs[0])}")

    # housekeeping
    if queue:
        queue.close()
    if submitter:
        submitter.close()
    
    print(f"Completed {nb_batches} batches of inference with batch size {args.bs} in {toc - tic:0.4f} seconds")
    print(f'Unet inference through-put: {args.bs * (1/((toc - tic) / nb_batches)):0.3f}/sec')


if __name__ == "__main__":
    run_async()
