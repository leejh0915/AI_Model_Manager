#Classification Test를 위한 간단한 ResNet 만들기
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from networks.backbone.resnet import ResNet
from networks.backbone.resnet import ResidualBlock

class SimpleResNet:
    def __init__(self):
        self.resnet = ResNet(ResidualBlock, [2, 2, 2])

    def set_networks(self):
        return self.resnet