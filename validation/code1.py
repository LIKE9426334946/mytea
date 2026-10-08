import torch
import torch.nn as nn

# 三个类别的 Logits
z = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)

# 真实类别为第 0 类
target = torch.tensor([0])

# 交叉熵损失
criterion = nn.CrossEntropyLoss()
loss = criterion(z, target)

# 反向传播
loss.backward()

print("Loss:", loss.item())
print("自动求导:", z.grad)

# 手动计算梯度
p = torch.softmax(z.detach(), dim=1)
y = torch.tensor([[1.0, 0.0, 0.0]])

print("手动求导:", p - y)