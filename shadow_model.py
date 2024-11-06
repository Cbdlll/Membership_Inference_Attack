import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# 模拟目标模型的 shadow model
class ShadowModel(nn.Module):
    def __init__(self):
        super(ShadowModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)  # 将通道数从32减小到16
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 12 * 12, 64)  # 将全连接层的神经元从128减小到64
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练shadow model
def train_shadow_model(shadow_model, train_loader, num_epochs=5):
    shadow_model.train()
    optimizer = optim.SGD(shadow_model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = shadow_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# 保存训练好的模型
def save_target_model(shadow_model, file_path):
    torch.save(shadow_model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

# 目标模型训练主程序
def train_and_save_shadow_model(shadow_train_loader, file_path='./shadow_model.pth'):
    shadow_model = ShadowModel()
    train_shadow_model(shadow_model, shadow_train_loader)
    save_target_model(shadow_model, file_path)

# 保存训练样本到文件
def save_subset_data(train_subset, file_path='shadow_train_data.pt'):
    # 保存图片数据和标签
    images = torch.stack([sample[0] for sample in train_subset])
    labels = torch.tensor([sample[1] for sample in train_subset])
    
    # 保存为 .pt 文件
    torch.save((images, labels), file_path)
    print(f"Subset data saved to {file_path}")

if __name__ == "__main__":
    
    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = torchvision.datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transform)
    
    # 随机选择1000个样本
    total_samples = len(train_data)
    random_indices = random.sample(range(total_samples), 1000)
    train_subset = Subset(train_data, random_indices)
    save_subset_data(train_subset, file_path='shadow_train_data.pt')
    
    shadow_train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    train_and_save_shadow_model(shadow_train_loader)