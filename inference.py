# 훈련시킨 모델을 가지고 inference할 class들에 대한 모듈
# 현재 trt,torch inference를 제공중이며 추후 tensorflow와 tflite 관련 클래스도 추가예정

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from time import time
import pycuda.driver as cuda
import pycuda.autoinit

import util.trt_func as trt_func

class trt_inference():
    def __init__(self, args, weight_path, dataset):
        self.mode = args.mode
        self.batch = args.batch_size
        self.ch_in = args.img_ch
        self.h_in = args.img_h_size
        self.w_in = args.img_w_size
        self.weight_path = weight_path
        self.dataset = dataset  # test_loader

    def run(self):
        if self.mode == 'classification':
            # batch, ch_in, h_in, w_in = self.batch_size, self.img_ch, self.img_h_size, self.img_w_size

            with trt_func.get_engine(self.weight_path) as engine, engine.create_execution_context() as context:
                buffers = trt_func.allocate_buffers(engine, batch_size=self.batch)
                # binding input shape
                context.set_binding_shape(0, (self.batch, self.ch_in, self.h_in, self.w_in))

                inputs, outputs, bindings, stream = buffers

                correct = 0
                total = 0

                start = time()
                for images, labels in self.dataset:
                    img = images.numpy().astype(np.float16)
                    lb = labels.numpy()
                    inputs[0].host = np.ascontiguousarray(img)  # ascontiguousarray는 빠르게 데이터를 부르는 역할을 함(결과 자체가 다르진 않음)

                    trt_outputs = trt_func.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs,
                                                        stream=stream)

                    predicted = trt_outputs[0].tolist().index(np.max(trt_outputs[0]))
                    total += labels.size(0)
                    correct += (predicted == lb).sum().item()

                    # print('numpy:{}'.format(trt_outputs[0]))
                    # print('list:{}'.format(trt_outputs[0].tolist()))
                    # print('pred: {}'.format(predicted))
                    # print('lb: {}'.format(lb))

                end = time()
                time_res = (end - start)

                print("time_res: ", time_res)
                print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

            return 0
        elif self.mode == 'detection':
            return print('준비중')
        elif self.mode == 'pose estimation':
            return print('준비중')
        else:
            return print('옳바른 정보가 아닙니다')


class torch_inference():
    def __init__(self, args, dataset, weight_path, network):
        self.mode = args.mode
        self.batch = args.batch_size
        self.ch_in = args.img_ch
        self.h_in = args.img_h_size
        self.w_in = args.img_w_size
        self.weight_path = weight_path
        self.dataset = dataset  # test_loader
        self.network = network  # import_network().set_networks()

    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.mode == 'classification':
            model = self.network
            model.to(device)
            model.load_state_dict(torch.load(self.weight_path))

            with torch.no_grad():
                correct = 0
                total = 0

                start = time()

                for images, labels in self.dataset:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                end = time()
                time_res = (end - start)

                print("time_res: ", time_res)
                print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

        elif self.mode == 'detection':
            return print('준비중')
        elif self.mode == 'pose estimation':
            return print('준비중')
        else:
            return print('옳바른 정보가 아닙니다')


class tf_inference():
    def __init__(self, mode, args, dataset, weight_path, network):
        self.mode = mode
        self.batch = args.batch_size
        self.ch_in = args.img_ch
        self.h_in = args.img_h_size
        self.w_in = args.img_w_size
        self.weight_path = weight_path
        self.dataset = dataset  # test_loader
        self.network = network  # import_network().set_networks()

    def run(self):
        pass


class tflite_inference():
    def __init__(self, mode, args, dataset, weight_path, network):
        self.mode = mode
        self.batch = args.batch_size
        self.ch_in = args.img_ch
        self.h_in = args.img_h_size
        self.w_in = args.img_w_size
        self.weight_path = weight_path
        self.dataset = dataset  # test_loader
        self.network = network  # import_network().set_networks()

    def run(self):
        pass