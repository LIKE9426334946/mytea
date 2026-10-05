import torch

# x: 1×3
x = torch.tensor([[2.0, 3.0, 4.0]], requires_grad=True)

# w: 3×1
w = torch.tensor([[5.0], [6.0], [7.0]], requires_grad=True)

# 矩阵乘法
y = x @ w

print("y =", y)

# 反向传播
y.backward()

print("x.grad =", x.grad)
print("w.grad =", w.grad)

