import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

# 标签映射字典
label_map = {'Red': 0, 'Green': 1}

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
        
        # 提取所有图片文件名和对应的标签
        self.img_names = [x[0] for x in self.img_labels]
        self.labels = torch.tensor([x[1] for x in self.img_labels], dtype=torch.long)  # 转为张量

    def __len__(self):
        # 返回数据集的大小
        return len(self.img_labels)

    def __getitem__(self, idx):
        # 获取图片的路径和标签
        img_name = self.img_names[idx]
        label = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 打开图像
        image = Image.open(img_path).convert("RGB")  # 转为RGB模式，如果是灰度图像可去掉 .convert("RGB")
        
        # 图像预处理
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def get_data_labels(self):
        """
        提供类似于 MNIST 的数据和标签张量
        """
        all_images = []
        all_labels = []

        for idx in range(len(self)):
            img, label = self[idx]
            all_images.append(img)
            all_labels.append(label)

        # 转换为Tensor
        data_tensor = torch.stack(all_images)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)

        return data_tensor, labels_tensor
