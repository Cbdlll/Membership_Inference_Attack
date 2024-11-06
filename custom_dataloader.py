import torch
from torch.utils.data import Dataset, DataLoader

# 定义一个Dataset类，用于加载目标训练集数据
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 返回指定索引的图片和标签
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# 加载保存的数据
def load_target_train_data(file_path='target_train_data.pt'):
    images, labels = torch.load(file_path, weights_only=True)
    return images, labels

# 加载数据并创建DataLoader
def create_dataloader(file_path='target_train_data.pt', batch_size=32, shuffle=True):
    images, labels = load_target_train_data(file_path)
    dataset = CustomDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# # 使用DataLoader
# dataloader = create_dataloader(file_path='target_train_data.pt', batch_size=32)
# print(len(dataloader))

# # 例子: 打印一个batch的数据
# for images, labels in dataloader:
#     print(images.shape, labels.shape)
#     break
