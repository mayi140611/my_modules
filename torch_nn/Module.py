# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py

import torch.nn as nn
import torch.nn.functional as F

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
    # 初始化
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        # nn.init.uniform_(self.fc.weight.data, -initrange, initrange)
        # nn.init.xavier_normal_(self.fc.weight.data)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
    
m = Model()

# print(m)
# Model(
#   (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1))
# )
# print(m.conv1) # Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
# for p in m.modules():
#     print(p)
# Model(
#   (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1))
# )
# Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
# Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1))

# 获取模型参数
# for param in m.parameters():
#     print(param) # 只打印参数
for name, param in m.named_parameters():
    if 'conv2' in name: # 冻结指定层的参数
        param.requires_grad = False
    print(f"{name}---{param}")
# print(type(m.conv1.weight)) # <class 'torch.nn.parameter.Parameter'>
# print(m.conv1.weight)
# Parameter containing:
# tensor([[[[-1.8957e-02, -5.5695e-02, -2.9702e-02,  6.9530e-02,  7.6027e-02],
#           [ 1.3081e-01, -7.4121e-02, -1.0782e-02,  7.8491e-02,  5.5318e-04],
#           [-1.4635e-01, -1.2151e-01,  4.2622e-02, -1.6320e-01, -1.8955e-01],
#           [-1.7521e-01,  2.0145e-02,  5.3978e-02,  1.5686e-01, -4.7546e-02],
#           [-7.5833e-02,  8.5170e-02, -3.6506e-03,  9.9738e-02,  1.4837e-01]]],
# ...
# [[[ 1.2419e-01, -1.5185e-01, -1.8889e-02,  4.2650e-02, -1.3262e-01],
#           [-5.5861e-03,  1.9538e-01,  8.4105e-02, -1.5714e-01, -9.9522e-02],
#           [-7.0490e-02,  1.3224e-01, -7.7383e-02, -4.0848e-02,  9.4387e-02],
#           [ 1.3202e-01,  3.3331e-02,  1.1398e-01, -1.8256e-01,  7.9965e-02],
#           [-4.4910e-02, -1.8403e-01, -2.1828e-02, -1.3707e-01,  9.9894e-02]]]],
#        requires_grad=True)
# print(type(m.conv1.weight.data)) # <class 'torch.Tensor'>
# print(m.conv1.weight.data)
# tensor([[[[ 0.1087,  0.0323, -0.1785, -0.1769, -0.0354],
#           [-0.0742, -0.1497, -0.1620,  0.0973,  0.0765],
#           [-0.0621, -0.0458,  0.0499,  0.1902, -0.1648],
#           [-0.0400, -0.0349, -0.1079,  0.1521, -0.1875],
#           [-0.0075,  0.1601, -0.0670, -0.1213,  0.0065]]],
#           ...
#         [[[-0.0107,  0.0257, -0.0960,  0.1278,  0.0331],
#           [-0.0300,  0.0251, -0.1870,  0.0668, -0.0535],
#           [-0.0939, -0.0177, -0.0013,  0.0771,  0.1842],
#           [-0.1441,  0.0489, -0.1984,  0.1464,  0.0206],
#           [-0.0960, -0.0564,  0.1263, -0.1930, -0.0291]]]])
# 获取bias
# print(m.conv1.bias)
# Parameter containing:
# tensor([ 0.0956,  0.0829, -0.0961,  0.0747,  0.1575, -0.0872,  0.1866,  0.1916,
#         -0.0012, -0.0183, -0.1568, -0.1120,  0.1609, -0.1030,  0.0879, -0.0663,
#         -0.1707, -0.0585,  0.0668, -0.0143], requires_grad=True)
# 权值初始化
# initrange = 0.5
# m.conv1.weight.data.uniform_(-initrange, initrange)
# m.conv1.bias.data.zero_()
# print(m.conv1.bias)
# Parameter containing:
# tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        requires_grad=True)








