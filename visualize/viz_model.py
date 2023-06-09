"""
Visualize the model
"""
import torch
import torch.nn as nn
from torchview import draw_graph

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pyt_unet_model import UNet

# usage:
input_shape = (1, 3, 1056, 160)
model = UNet(input_shape)

model_graph = draw_graph(model,
                         input_size=input_shape)
model_graph.visual_graph.render('unet-pyt-model')


model_graph_2 = draw_graph(model,
                         input_size=input_shape)

model_graph_2.resize_graph(scale=0.5)

model_graph_2.visual_graph.render('unet-half-size-pyt-model')

