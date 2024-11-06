# target_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

class TargetModel(nn.Module):
    def __init__(self):
        super(TargetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# 训练目标模型的函数
def train_target_model(target_model, train_loader, num_epochs=5):
    target_model.train()
    optimizer = optim.SGD(target_model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = target_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")


# 保存训练好的模型
def save_target_model(target_model, file_path):
    torch.save(target_model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


# 目标模型训练主程序
def train_and_save_target_model(train_loader, file_path='./target_model.pth'):
    target_model = TargetModel()
    train_target_model(target_model, train_loader)
    save_target_model(target_model, file_path)

# 保存训练样本到文件
def save_subset_data(train_subset, file_path='target_train_data.pt'):
    # 保存图片数据和标签
    images = torch.stack([sample[0] for sample in train_subset])
    labels = torch.tensor([sample[1] for sample in train_subset])
    
    # 保存为 .pt 文件
    torch.save((images, labels), file_path)
    print(f"Subset data saved to {file_path}")

if __name__ == "__main__":
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = torchvision.datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transform)
    # 只使用前1000个样本进行训练
    subset_indices = list(range(1000))  # 选择前1000个样本的索引
    train_subset = Subset(train_data, subset_indices)
    save_subset_data(train_subset, file_path='target_train_data.pt')
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    train_and_save_target_model(train_loader)