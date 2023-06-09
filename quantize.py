#!/usr/bin/env python3

"""
Needs the imagnet 'val' folder with the structure retained 
and a few images for the calibration functions to work
"""
import sys

import onnx
import torch
import torchvision
from torchvision import transforms
import tqdm

from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod

from pudb import set_trace

import argparse

parser = argparse.ArgumentParser(
                    prog='convert_2_onnx',
                    description='converts to onnx',
                    epilog='look for file with .onnx suffix')

parser.add_argument('--bs', required=True, type=int)
args = parser.parse_args()
print(f'Requested batch size: {args.bs}')

BATCH_SIZE = args.bs 
def create_quantized_dfg():
    model = onnx.load_model("pyt_unet.onnx")
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((1056, 160)),
         transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )   

    calibration_dataset = torchvision.datasets.ImageNet('../',
                                                        split='val',
                                                        transform=preprocess)
    print(f'shape of calibration data: {calibration_dataset}')
    
    calibration_dataloader = torch.utils.data.DataLoader(calibration_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)
    model = optimize_model(model)
    # optimize_model(model, input_shapes={"input": [BATCH_SIZE, 3, 1050, 160]}) 
    calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)

    for calibration_data, _ in tqdm.tqdm(calibration_dataloader,
                                         desc="Calibration",
                                         unit="images",
                                         mininterval=0.5):
        print(f'calibration data shape: {calibration_data.shape}')
        # calibration_data = torch.permute(calibration_data, (0, 2, 3, 1))
        # print(f'calibration data shape: {calibration_data.shape}')

        # set_trace()
        calibrator.collect_data([[calibration_data.numpy()]])

    ranges = calibrator.compute_range()
    print(f'ranges" {ranges}')
    # set_trace()

    model_quantized = quantize(model, ranges)

    with open("unet_model_quantized.dfg", "wb") as f:
        f.write(bytes(model_quantized))


if __name__ == "__main__":
    sys.exit(create_quantized_dfg())    
