import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import argparse
import random
import math
import numpy as np
import torch
import os
import sys
from torch import nn, optim, autograd
from torchvision import datasets

label_map = {'Red': -1, 'Green': 1, 'Yellow': 0}
class CustomDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        """
        :param img_dir: 图片所在文件夹路径
        :param label_file: 标签txt文件路径
        :param transform: 数据转换方式（如归一化，调整大小等）
        """
        self.img_dir = img_dir
        self.label_file = label_file
        self.transform = transform
        
        # 读取标签文件
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # 假设标签文件格式：每行是图片的文件名和对应的标签
        self.img_labels = [(line.split()[0], label_map[line.split()[1]]) for line in lines]

    def __len__(self):
        # 数据集大小
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 打开图像
        image = Image.open(img_path).convert("RGB")  # 如果是灰度图像，可以去掉 .convert("RGB")
        
        # 图像预处理
        if self.transform:
            image = self.transform(image)
        
        return image, label

# # 数据增强/预处理
# transform = transforms.Compose([
#     transforms.Resize((28, 28)),  # 假设图片大小调整为28x28，小一点训练快
#     transforms.ToTensor(),        # 转换为Tensor
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化，标准化处理
# ])

# # 创建数据集和数据加载器
# img_dir = "dataset/trafficlight_data_sample"  # 图片目录
# label_file = "dataset/trafficlight_data_sample/labels.txt"  # 标签文件路径
# dataset = CustomDataset(img_dir=img_dir, label_file=label_file, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



# mnist = datasets.MNIST('~/datasets/mnist', train=True, download=False) # download

# # 遍历数据加载器中的数据
# for images, labels in dataloader:
#     print(images.size(), labels.size())

