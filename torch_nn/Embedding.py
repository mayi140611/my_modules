from torch import nn

# 定义
# torch.nn.Embedding(
#     num_embeddings: int, # size of the dictionary of embeddings
#     embedding_dim: int, 
#     padding_idx: Optional[int] = None, 
#     max_norm: Optional[float] = None, 
#     norm_type: float = 2.0, 
#     scale_grad_by_freq: bool = False, 
#     sparse: bool = False, 
#     _weight: Optional[torch.Tensor] = None
# )
# 重要参数
# padding_idx (int, optional) – If specified, the entries at padding_idx do not contribute to the gradient; 
#     therefore, the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”. 
#     For a newly constructed Embedding, the embedding vector at padding_idx will default to all zeros, 
#     but can be updated to another value to be used as the padding vector.




# new
# embedding = nn.Embedding(10, 3)
# embedding.weight.data
# tensor([[ 0.4336,  0.6459, -0.5624],
#         [ 0.3680,  1.1747, -0.9030],
#         [ 0.5488, -1.0322,  0.0983],
#         [-0.9508,  0.0124,  0.9986],
#         [-0.5253, -1.6486,  0.9243],
#         [-1.2660,  0.4701, -0.0711],
#         [ 0.2583,  0.8232,  0.1713],
#         [ 0.6192,  1.5003,  2.2059],
#         [-0.5282,  1.6544,  0.1958],
#         [-0.2933,  0.6812,  1.1026]])

# a batch of 2 samples of 4 indices each
# input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# embedding(input)
# tensor([[[ 0.3680,  1.1747, -0.9030],
#          [ 0.5488, -1.0322,  0.0983],
#          [-0.5253, -1.6486,  0.9243],
#          [-1.2660,  0.4701, -0.0711]],

#         [[-0.5253, -1.6486,  0.9243],
#          [-0.9508,  0.0124,  0.9986],
#          [ 0.5488, -1.0322,  0.0983],
#          [-0.2933,  0.6812,  1.1026]]], grad_fn=<EmbeddingBackward0>)

padding_idx = 0
embedding = nn.Embedding(3, 4, padding_idx=padding_idx)
# print(embedding.weight.data)
# tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.6655,  0.8374, -1.1194,  0.0337],
#         [ 0.8482,  0.2555, -0.3121, -0.0693]])

# class LlamaModel(LlamaPreTrainedModel):
# self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

# class QWenModel(QWenPreTrainedModel):
# self.wte = nn.Embedding(self.vocab_size, self.embed_dim)

# class InternLM2Model(InternLM2PreTrainedModel):
# self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

# 参数初始化
        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()