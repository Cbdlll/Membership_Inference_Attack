import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from attack_model import AttackModel, is_image_in_image_data
from target_model import TargetModel

# 评估攻击模型
def evaluate_attack_model(attack_model, target_model, eval_data_loader, target_train_data):
    attack_model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    
    # 关闭梯度计算
    with torch.no_grad():
        for inputs, _ in eval_data_loader:
            # 获取目标模型的logits
            target_model_logits = target_model(inputs)
            
            # 获取模型的预测输出
            outputs = attack_model(F.softmax(target_model_logits, dim=1))
            
            # 对攻击模型的输出应用阈值（假设输出是概率）
            predicted = (outputs.squeeze() > 0.5).float()  # 阈值0.5判断属于目标数据集或非目标数据集
            
            # 获取对应的真实标签
            labels = torch.tensor([is_image_in_image_data(input, target_train_data) for input in inputs])
            
            # 统计正确预测的数量
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total * 100 # 计算准确率
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy

# 加载训练好的攻击模型
def load_attack_model(file_path='./attack_model.pth'):
    attack_model = AttackModel()
    attack_model.load_state_dict(torch.load(file_path, weights_only=True))  # 加载保存的模型权重
    attack_model.eval()
    print(f"Attack model loaded from {file_path}")
    return attack_model

# 加载训练好的目标模型
def load_target_model(file_path='./target_model.pth'):
    target_model = TargetModel()
    target_model.load_state_dict(torch.load(file_path, weights_only=True))  # 加载保存的模型权重
    target_model.eval()
    print(f"Target model loaded from {file_path}")
    return target_model

if __name__ == "__main__":
    
    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = torchvision.datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transform)
 
    # 随机选择1000个样本用作evaluate
    total_samples = len(train_data)
    eval_random_indices = random.sample(range(total_samples), 1000)
    eval_subset = Subset(train_data, eval_random_indices)
    eval_data_loader = DataLoader(eval_subset, batch_size=64, shuffle=True)
    
    target_train_data, _ = torch.load('target_train_data.pt', weights_only=True)
    
    # 加载攻击模型
    attack_model = load_attack_model()
    # 评估攻击模型
    target_model = load_target_model()
    
    evaluate_attack_model(attack_model, target_model, eval_data_loader, target_train_data)