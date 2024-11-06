import torch

from target_model import TargetModel


model = TargetModel()
model.load_state_dict(torch.load('./target_model.pth', weights_only=True))
    
# 打印所有权重和偏置的形状和具体值
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
    print(f"{name}: {param.data}")