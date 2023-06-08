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

BATCH_SIZE = 1
def create_quantized_dfg():
    set_trace()
    model = onnx.load_model("pyt_unet.onnx")
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((1056, 160)),
         transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )   

    calibration_data = torchvision.datasets.ImageNet('../',
                                                     split='val',
                                                     transform=preprocess)
    calibration_dataloader = torch.utils.data.DataLoader(calibration_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)
    model = optimize_model(model)

    calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)

    for calibration_data, _ in tqdm.tqdm(calibration_dataloader,
                                         desc="Calibration",
                                         unit="images",
                                         mininterval=0.5):
        print(f'calibration data shape: {calibration_data.shape}')
        # calibration_data = torch.permute(calibration_data, (0, 2, 3, 1))
        # print(f'calibration data shape: {calibration_data.shape}')
        calibrator.collect_data([[calibration_data.numpy()]])

    ranges = calibrator.compute_range()
    print(f'ranges" {ranges}')
    set_trace()

    model_quantized = quantize(model, ranges)

    with open("unet_model_quantized.dfg", "wb") as f:
        f.write(bytes(model_quantized))


if __name__ == "__main__":
    sys.exit(create_quantized_dfg())    
