# 훈련시킨 모델을 가지고 inference를 진행합니다
# 두 모델의 inference 결과를 비교하기도 합니다
# Detection, Keypoint, Classification 등 다양한 알고리즘에 inference가 가능하도록 설계할 예정입니다

import argparse
from util.util import load_module_func

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # select train or inference mode
    parser.add_argument('--inference_mode', type=bool, default=True, help='두 모델을 비교하는 모드로 진행합니다')
    parser.add_argument('--train_mode', type=bool, default=False, help='두 모델을 비교하는 모드로 진행합니다')

    # select algorithm mode(classification, detection, pose estimation, segmentation, etc..)
    parser.add_argument('--algorithm_mode', default='classification', help='실행시키고자 하는 알고리즘 방식을 선택합니다')

    # inference options
    parser.add_argument('--network', default='algorithm_list.simple_resnet.SimpleResNet', help='네트워크를 불어옵니다')
    parser.add_argument('--weight_path',
                        default='/home/jhyeok.lee/workspace/prevmodels/model_compression/resnet_ori_fp16.trt',
                        help='해당되는 네트워크의 모델 웨이트를 로드합니다')
    parser.add_argument('--inference', default='inference.trt_inference', help='해당되는 API의 inference를 호출해 관련 내용들을 측정합니다')

    # compare inference options
    parser.add_argument('--compare_mode', type=bool, default=True, help='두 모델을 비교하는 모드로 진행합니다')
    parser.add_argument('--compare_network', default='algorithm_list.simple_resnet.SimpleResNet',
                        help='비교할 네트워크를 불러옵니다')
    parser.add_argument('--compare_weight_path',
                        default='/home/jhyeok.lee/workspace/prevmodels/model_compression/resnet.pt',
                        help='비교할 네트워크의 모델 웨이트를 로드합니다')
    parser.add_argument('--compare_inference', default='inference.torch_inference',
                        help='비교할 API의 inference를 호출해 관련 내용들을 측정합니다')

    # select dataset
    parser.add_argument('--select_dataset', default='dataset.cifar10.cifar10',
                        help='데이터셋 파일을 불러옵니다')

    # input data Info
    parser.add_argument('--img_h_size', default=32, help='이미지의 높이값 정보')
    parser.add_argument('--img_w_size', default=32, help='이미지의 너비값 정보')
    parser.add_argument('--img_ch', default=3, help='이미지의 채널값 정보')
    parser.add_argument('--batch_size', default=1, help='배치 사이즈의 크기 정보')

    args = parser.parse_args()
    import_network = load_module_func(args.network)  # network_import
    import_dataset = load_module_func(args.select_dataset)  # dataset_import
    import_inference = load_module_func(args.inference)  # inference_mode_import

    # test_dataset, test_loader = import_dataset().set_test_dataset()
    _, test_loader = import_dataset().set_test_dataset()

    if args.inference_mode:
        if args.compare_mode:
            import_compare_inference = load_module_func(args.compare_inference)  # inference_mode_import

            import_inference(mode='classification', args=args, weight_path=args.weight_path, dataset=test_loader).run()
            import_compare_inference(mode='classification', args=args, weight_path=args.compare_weight_path,
                                     dataset=test_loader, network=import_network().set_networks()).run()
            print('Compare inference Complete')
        else:
            import_inference(mode='classification', args=args, weight_path=args.weight_path, dataset=test_loader).run()
            print('inference Complete')
    else:
        print('training 관련 내용 준비중')