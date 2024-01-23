# Module
[源代码](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py)
[说明文档API](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
Base class for all neural network modules. 定义了一些基本操作。要学会熟练使用！
## 基本操作
### 如何获取模型的结构
有哪些layer？每个layer的类型等

### 获取某层的参数
m.conv1.weight

for name, param in m.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

m.state_dict() # Returns a dictionary containing references to the whole state of the module.

for param in model.parameters():
    print(type(param), param.size())
### 初始化某层的参数？
```
# 权值初始化
initrange = 0.5
m.conv1.weight.data.uniform_(-initrange, initrange)
m.conv1.bias.data.zero_()
```
### 冻结某层的参数
```
for name, param in m.named_parameters():
    if 'conv2' in name: # 冻结指定层的参数
        param.requires_grad = False
    print(f"{name}---{param}")
```
### save & load
torch.save(model.state_dict(), "model.pth")

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

# Optimizer
https://pytorch.org/docs/stable/optim.html

Optimizer负责维护优化器的状态及更新参数
# [Embedding层](./Embedding.py)
https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding

作用就是给输入的tensor扩充一个维度，如(batch_size, seq_len, token_ids) => (batch_size, seq_len, token_ids, embedding_dim)

在nlp的理解中就是原本一个token是用一个数字表示的，现在改为用一个向量表示，所蕴含的信息更加丰富。