# convert가 잘 이루어졌는지 비교하고
# 초기 버전은 torch와 ONNX,TensorRT만 적용하며 args를 이용해 비교할 내용들을 다룰 수 있도록 설계할 예정입니다.
# 추후에는 arg값을 사용해 Tensorflow, Keras, TFLite 등의 API도 적용가능하도록 설계할 예정입니다

import os
import numpy as np
import torch.nn as nn
import torch.onnx
from torchvision import models
import onnx
from onnx import shape_inference
import onnx.numpy_helper as numpy_helper

from networks.backbone.resnet import ResNet
from networks.backbone.resnet import ResidualBlock


def compare_two_array(actual, desired, layer_name, rtol=1e-7, atol=0):
    # Reference : https://gaussian37.github.io/python-basic-numpy-snippets/
    flag = False
    try:
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
        print(layer_name + ": no difference.")
    except AssertionError as msg:
        print(layer_name + ": Error.")
        print(msg)
        flag = True
    return flag


# parameters
channel = 3
height = 32
width = 32
onnx_path = '/home/jhyeok.lee/workspace/prevmodels/model_compression/resnet_ori.onnx'

# 1 사용할 딥러닝 네트워크를 불러온 뒤 평가 모드로 설정합니다.
# Device configuration
device = torch.device('cpu')
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
model.load_state_dict(torch.load('/home/jhyeok.lee/workspace/prevmodels/model_compression/resnet.pt'))
x = torch.ones(1, channel, height, width, dtype=torch.float32)

# 3 생성한  onnx 모델을 다시 블루어와서 torch 모델과 onnx 모델의 weight를 비교합니다.
# 입력 받은 onnx 파일 경로를 통해 onnx 모델을 불러옵니다.
onnx_model = onnx.load(onnx_path)

# onnx 모델의 정보를 layer 이름 : layer값 기준으로 저장합니다.
onnx_layers = dict()
for layer in onnx_model.graph.initializer:
    onnx_layers[layer.name] = numpy_helper.to_array(layer)

# torch 모델의 정보를 layer 이름 : layer값 기준으로 저장합니다.
torch_layers = {}
for layer_name, layer_value in model.named_modules():
    torch_layers[layer_name] = layer_value

# onnx와 torch 모델의 성분은 1:1 대응이 되지만 저장하는 기준이 다릅니다.
# onnx와 torch의 각 weight가 1:1 대응이 되는 성분만 필터합니다.
onnx_layers_set = set(onnx_layers.keys())
# onnx 모델의 각 layer에는 .weight가 suffix로 추가되어 있어서 문자열 비교 시 추가함
torch_layers_set = set([layer_name + ".weight" for layer_name in list(torch_layers.keys())])
filtered_onnx_layers = list(onnx_layers_set.intersection(torch_layers_set))

difference_flag = False
for layer_name in filtered_onnx_layers:
    onnx_layer_name = layer_name
    torch_layer_name = layer_name.replace(".weight", "")
    onnx_weight = onnx_layers[onnx_layer_name]
    torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
    flag = compare_two_array(onnx_weight, torch_weight, onnx_layer_name)
    difference_flag = True if flag == True else False

# 4 onnx 모델에 기존 torch 모델과 다른 weight가 있으면 전체 update를 한 후 새로 저장합니다.
if difference_flag:
    print("update onnx weight from torch model.")
    for index, layer in enumerate(onnx_model.graph.initializer):
        layer_name = layer.name
        if layer_name in filtered_onnx_layers:
            onnx_layer_name = layer_name
            torch_layer_name = layer_name.replace(".weight", "")
            onnx_weight = onnx_layers[onnx_layer_name]
            torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
            copy_tensor = numpy_helper.from_array(torch_weight, onnx_layer_name)
            onnx_model.graph.initializer[index].CopyFrom(copy_tensor)

    # print("save updated onnx model.")
    # onnx_new_path = os.path.dirname(os.path.abspath(onnx_path)) + os.sep + "updated_" + os.path.basename(onnx_path)
    # onnx.save(onnx_model, onnx_new_path)

# 5 최종적으로 저장된 onnx 모델을 불러와서 shape 정보를 추가한 뒤 다시 저장합니다.
# if difference_flag:
#     onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_new_path)), onnx_new_path)
# else:
#     onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)


# Reference https://gaussian37.github.io/dl-pytorch-deploy/