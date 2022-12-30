import os, sys
import torch.onnx
import argparse

from torchsummaryX import summary

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.util import load_module_func

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # convert options
    parser.add_argument('--load_weight', default='/home/jhyeok.lee/workspace/prevmodels/model_compression/resnet.pt', help='path to load model')
    parser.add_argument('--save_weight', default='./test_res.onnx', help='path to save model')
    parser.add_argument('--network', default='algorithm_list.simple_resnet.SimpleResNet', help='path to save model')
    parser.add_argument('--convert_method', default='onnx2', help='path to save model')

    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import_network = load_module_func(args.network)
    res = import_network().set_networks()

    model = res.to(device)
    model.load_state_dict(torch.load(args.load_weight))
    x = torch.ones((1, 3, 32, 32)).cuda()
    summary(model, x)

    #아래 부분들은... if문 이외의 다른 방식으로 변환 하는 것도 고민중...
    if args.convert_method == 'torch2onnx':
        torch.onnx.export(model,               # 실행될 모델
                        x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                        args.save_weight,   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                        export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                        opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                        do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                        input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                        output_names = ['output'], # 모델의 출력값을 가리키는 이름
                        dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                    'output' : {0 : 'batch_size'}})
    elif args.convert_method == 'onnx2trt':
        print('준비중....')
    else:
        print('관련된 변환 정보가 없습니다....')


    print('finish!!!!')