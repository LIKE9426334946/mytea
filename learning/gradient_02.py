import torch

# x: 1×3
x = torch.arange(6, dtype=torch.float).reshape(2, 3)

# w: 3×1
w = torch.tensor([[5.0, 4.0], [6.0, 3.0], [7.0, 2.0]], requires_grad=True)

for epoch in range(100):
    w.grad = None
    # 矩阵乘法
    y = x @ w
    y.retain_grad()

    # 反向传播
    loss = y.sum()
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad

    # print("x.grad =", x.grad)

print(w)
print(loss)