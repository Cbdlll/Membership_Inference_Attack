⚠️**WARNING:** 由于使用数据集数量非常少，所以以下运行结果 **不可复现!** ⚠️

⚠️**WARNING:** Due to the very small dataset size, the following results are **NOT reproducible!** ⚠️

# 成员推理攻击

**成员推理攻击**（Membership Inference Attack，MIA）中，攻击者试图判断一个给定的数据点是否存在于目标模型的训练集中。为了进行这样的攻击，通常需要构建三个关键模型：**Target Model**、**Shadow Model** 和 **Attack Model**。

**目标模型（Target Model）**：攻击者试图推测其训练数据成员，通常是黑盒的，无法直接访问。

**影子模型（Shadow Model）**：用来模拟目标模型的行为，通常在与目标模型相似的训练数据集上进行训练。影子模型提供的输出用于训练攻击模型。

**攻击模型（Attack Model）**：利用影子模型的输出进行训练，最终执行成员推理攻击，判断一个数据点是否属于目标模型的训练集。

## 工作流程

1. **训练影子模型**：攻击者收集一个与目标模型相似的训练集，并训练一个影子模型。影子模型的目标是尽可能模仿目标模型的行为。
2. **生成攻击数据**：通过将影子模型应用于训练数据和非训练数据，收集影子模型的输出（通常是 logits 或 softmax 输出）。这些输出将作为攻击模型的输入特征。
3. **训练攻击模型**：使用影子模型的输出数据来训练攻击模型，攻击模型学习如何基于模型的输出区分训练数据和非训练数据。
4. **评估攻击模型**：一旦攻击模型训练完成，就可以使用它来推断目标模型的训练数据成员。在攻击过程中，攻击模型将分析目标模型对某个数据点的输出，并判断该点是否属于目标模型的训练集。

# target_model

构造一个简单的target_model完成训练并保存。

```python
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
    
if __name__ == "__main__":
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = torchvision.datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transform)
    # 只使用前1000个样本进行训练
    subset_indices = list(range(1000))  # 选择前1000个样本的索引
    train_subset = Subset(train_data, subset_indices)
    save_subset_data(train_subset, file_path='target_train_data.pt')
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    train_and_save_target_model(train_loader)
```

```shell
wjg@14x:~/project/Membership_Inference_Attack$ python target_model.py 
Subset data saved to target_train_data.pt
Epoch 1, Loss: 2.076657697558403
Epoch 2, Loss: 1.0831771083176136
Epoch 3, Loss: 0.5170555431395769
Epoch 4, Loss: 0.4522471185773611
Epoch 5, Loss: 0.34391375351697206
Model saved to ./target_model.pth
```

# shadow_model

构造一个简单的shadow_model（与target_model结构不同），使用同分布的训练数据，完成训练并保存。

```python
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
```

```shell
wjg@14x:~/project/Membership_Inference_Attack$ python shadow_model.py 
Subset data saved to shadow_train_data.pt
Epoch 1, Loss: 2.171294055879116
Epoch 2, Loss: 1.398248229175806
Epoch 3, Loss: 0.6463910657912493
Epoch 4, Loss: 0.4396038670092821
Epoch 5, Loss: 0.3404361121356487
Model saved to ./shadow_model.pth
```

# attack_model

构造一个攻击模型，使用攻击数据集进行训练，完成训练并保存。

攻击数据集由shadow_model生成，存储有shadow_model对数据的预测概率分布以及是否属于训练集（shadow_model训练集）的label。

```python
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
```

```shell
wjg@14x:~/project/Membership_Inference_Attack$ python attack_model.py 
Shadow model loaded from ./shadow_model.pth
Epoch 1/5, Loss: 0.7187
Epoch 2/5, Loss: 0.7124
Epoch 3/5, Loss: 0.7062
Epoch 4/5, Loss: 0.7000
Epoch 5/5, Loss: 0.6939
```

# evaluate

评估攻击模型的性能，使用target_model对数据的预测概率分布作为输入给到attack_model，attack根据输入判断该条数据是否属于target_model的训练集。

```python
# 评估攻击模型
def evaluate_attack_model(attack_model, target_model, data_loader, target_train_data):
    attack_model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    true_positive = 0
    false_negative = 0
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            # 获取目标模型的logits
            target_model_logits = target_model(inputs)
            
            # 获取攻击模型的预测输出
            outputs = attack_model(F.softmax(target_model_logits, dim=1))
            
            # 对攻击模型的输出应用阈值（假设输出是概率）
            predicted = (outputs.squeeze() > 0.5).float()  # 阈值0.5判断属于目标数据集或非目标数据集
            
            # 获取对应的真实标签
            labels = torch.tensor([is_image_in_image_data(input, target_train_data) for input in inputs])
            
            # 统计真正类和假负类
            true_positive += ((predicted == 1) & (labels == 1)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()
            
            # 统计正确预测的数量
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total * 100 
    recall = (true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0) * 100
    
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    print(f"Recall: {recall:.2f}%")
    
    return accuracy, recall
```

```shell
wjg@14x:~/project/Membership_Inference_Attack$ python eval.py 
Attack model loaded from ./attack_model.pth
Target model loaded from ./target_model.pth
eval on target_train_data:
Evaluation Accuracy: 44.50%
Recall: 44.50%
eval on random_eval_data:
Evaluation Accuracy: 59.10%
Recall: 62.50%
```

