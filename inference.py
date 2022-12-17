# Detection, Keypoint, Classification 등 다양한 알고리즘에 inference가 가능하도록 설계할 예정입니다

import argparse
import torch
from util.util import load_module_func

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # inference options
    parser.add_argument('--network', default='algorithm_list.simple_resnet.SimpleResNet', help='path to load model')
    parser.add_argument('--weight_path', default='/home/jhyeok.lee/workspace/prevmodels/model_compression/resnet.pt',
                        help='path to load model')

    # compare inference options
    parser.add_argument('--compare_mode', type=bool, default=False, help='compare to models')
    parser.add_argument('--compare_network', default='algorithm_list.simple_resnet.SimpleResNet',
                        help='path to load model')
    parser.add_argument('--compare_weight_path',
                        default='/home/jhyeok.lee/workspace/prevmodels/model_compression/resnet.pt',
                        help='path to load model')

    # select dataset
    parser.add_argument('--select_dataset', default='dataset.cifar10.cifar10', help='path to load model')

    args = parser.parse_args()
    import_network = load_module_func(args.network)  # network_import
    import_dataset = load_module_func(args.select_dataset)  # dataset_import

    test_dataset, test_loader = import_dataset().set_test_dataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.compare_mode:
        # 해당 모델들을 평가 및 비교합니다
        model = import_network().set_networks()
        model.to(device)
        model.load_state_dict(torch.load(args.weight_path))

        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

        import_compare_network = load_module_func(args.compare_network)  # network_import

        compare_model = import_compare_network().set_networks()
        compare_model.to(device)
        compare_model.load_state_dict(torch.load(args.compare_weight_path))

        with torch.no_grad():
            compare_correct = 0
            compare_total = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = compare_model(images)
                _, predicted = torch.max(outputs.data, 1)
                compare_total += labels.size(0)
                compare_correct += (predicted == labels).sum().item()

            print(
                'Accuracy of the compare_model on the test images: {} %'.format(100 * compare_correct / compare_total))

    else:
        # 해당 모델을 평가합니다
        model = import_network().set_networks()
        model.to(device)
        model.load_state_dict(torch.load(args.weight_path))

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))