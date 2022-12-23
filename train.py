# # ---------------------------------------------------------------------------- #
# # An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# # See section 4.2 for the model architecture on CIFAR-10                       #
# # Some part of the code was referenced from below                              #
# # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# # ---------------------------------------------------------------------------- #
#
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
#
# from algorithm_list.backbone.resnet import ResNet
# from algorithm_list.backbone.resnet import ResidualBlock
#
# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Hyper-parameters
# num_epochs = 80
# learning_rate = 0.001
#
# # Image preprocessing modules
# transform = transforms.Compose([
#     transforms.Pad(4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32),
#     transforms.ToTensor()])
#
# # CIFAR-10 dataset
# train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
#                                              train=True,
#                                              transform=transform,
#                                              download=True)
#
# test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
#                                             train=False,
#                                             transform=transforms.ToTensor())
#
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=100,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=100,
#                                           shuffle=False)
#
#
# #해당 모델들을 훈련 및 평가합니다
# model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
#
#
# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # For updating learning rate
# def update_lr(optimizer, lr):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
# # Train the model
# total_step = len(train_loader)
# curr_lr = learning_rate
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()