from torch import optim


# 优化器初始化
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
# 指定每一层的学习率
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)

# Taking an optimization step
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward() # 这一步求出每个参数的梯度 grad
    optimizer.step() # 这一步更新 参数值 即：new_param = old_param - lr * grad