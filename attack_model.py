import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from shadow_model import ShadowModel

# 判断是否包含图像
def is_image_in_image_data(image, image_data):
    for stored_image in image_data:
        if torch.equal(stored_image, image):
            return 1  # 找到匹配的图片，返回 1
    return 0  # 没有找到匹配的图片，返回 0


# 生成攻击数据集（成员推理攻击数据集）
def generate_attack_data(shadow_model, attack_train_data_loader, shadow_train_data):
    attack_data = []
    attack_labels = []
    
    shadow_model.eval()
    
    with torch.no_grad():
        for inputs, _ in attack_train_data_loader:
            # 获取影子模型的预测logits
            outputs = shadow_model(inputs)
            
            # 存储影子模型的softmax(logits)，并确保dim=1
            softmax_outputs = F.softmax(outputs, dim=1)  # 确保沿着类维度应用 softmax
            
            # 存储每个批次的 softmax 输出和标签
            attack_data.append(softmax_outputs)
            
            # 检查该输入是否属于训练集
            labels = torch.tensor([is_image_in_image_data(input, shadow_train_data) for input in inputs])
            attack_labels.append(labels)

    # 将所有批次的数据和标签堆叠成一个大张量
    attack_data = torch.cat(attack_data, dim=0)  # 沿着第0维（batch维度）进行堆叠
    attack_labels = torch.cat(attack_labels, dim=0)  # 同样沿着第0维堆叠标签
    
    return attack_data, attack_labels


# 攻击模型（用于成员推理攻击）
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # 输入维度是模型输出的概率（10个类别的最大概率）
        self.fc2 = nn.Linear(64, 1)  # 输出维度为1（成员推理的结果）
        self.sigmoid = nn.Sigmoid()  # 用于二分类（属于训练集或非训练集）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 训练攻击模型
def train_attack_model(attack_model, attack_data, attack_labels, num_epochs=5):
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()  # 二元交叉熵损失
    
    attack_model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # 前向传播
        outputs = attack_model(attack_data)
        # 计算损失
        loss = loss_fn(outputs.squeeze(), attack_labels.float())
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    torch.save(attack_model.state_dict(), 'attack_model.pth')

# 加载训练好的影子模型
def load_shadow_model(file_path='./shadow_model.pth'):
    shadow_model = ShadowModel()
    shadow_model.load_state_dict(torch.load(file_path, weights_only=True))  # 加载保存的模型权重
    shadow_model.eval()
    print(f"Shadow model loaded from {file_path}")
    return shadow_model

if __name__ == "__main__":
    
    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = torchvision.datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transform)
    
    # 随机选择1000个样本用作attack model训练
    total_samples = len(train_data)
    attack_random_indices = random.sample(range(total_samples), 1000)
    attack_subset = Subset(train_data, attack_random_indices)
    attack_train_data_loader = DataLoader(attack_subset, batch_size=64, shuffle=True)
    
    # 随机选择1000个样本用作evaluate
    total_samples = len(train_data)
    eval_random_indices = random.sample(range(total_samples), 1000)
    eval_subset = Subset(train_data, eval_random_indices)
    eval_train_data_loader = DataLoader(eval_subset, batch_size=64, shuffle=True)
    
    # 加载影子模型的训练数据
    shadow_train_data, _ = torch.load('shadow_train_data.pt', weights_only=True)

    shadow_model = load_shadow_model('./shadow_model.pth')
    
    attack_data, attack_labels = generate_attack_data(shadow_model, attack_train_data_loader, shadow_train_data)
    
    attack_model = AttackModel()
    train_attack_model(attack_model, attack_data, attack_labels)
    