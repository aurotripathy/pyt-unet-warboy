from pyt_unet_model import UNet
import os
import logging
import torch
from pudb import set_trace

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

batch_size = 1
# Example usage:
input_shape = (batch_size, 3, 1056, 160)
model = UNet(input_shape)
from torchsummary import summary
summary(model, input_size=(3, 1056, 160))

# Input to the model
x = torch.randn(batch_size, 3, 1056, 160, requires_grad=True)
set_trace()
torch_model = UNet(input_shape)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "pyt_unet.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
