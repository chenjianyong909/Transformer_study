import torch
import torch.nn as nn
import math
# 可学习位置编码
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        # 输入的参数确定的原理跟sinusoidal是一样的
        # PE最终就是要跟输入的X进行相加，所以从X的维度去理解就可以了。
        # X的维度是(B, L, D)，其中B是batch size，L是序列长度，D是词向量维度。
        # 所以PE的维度也应该是(B, L, D)，但是每一个batch可以使用相同的PE。
        # 位置记录的是单条句子内部token的先后顺序，相同位置序号在不同句子里含义一样，所以batch里所有样本可以共用一套PE。
        # 最终的PE只要（L,D）就可以了
        super().__init__()
        if d_model <= 0 or max_len <= 0:
            raise ValueError('d_model 与 max_len 必须为正整数。')
        # 这样子出了__init__()，就可以在 forward() 里直接使用 self.d_model、self.max_len。
        # 没有self接收的话，函数结束后就会被销毁，forward()里就无法访问了。
        self.d_model = d_model
        self.max_len = max_len
        # nn.Embedding 的本质是拥有一个可学习的 weight 参数矩阵。
        # num_embeddings：词表大小，这里就是位置表的长度 max_len；
        # embedding_dim：词向量维度，这里就是位置向量维度 d_model。
        # 注意：可学习PE的weight是待训练参数，和固定正弦PE（注册buffer）不一样！
        self.pe = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)

    def forward(self, x, start_pos: int = 0):
        """
        x.shape = (B, L, D)
        start_pos：当前输入序列的起始位置，默认从 0 开始。
        """
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim != self.d_model:
            raise ValueError(f'输入的词向量维度 {embed_dim} 与初始化的 d_model {self.d_model} 不匹配。')
        if start_pos + seq_len > self.max_len:
            raise ValueError('当前 Token 的位置超出了预设的 max_len。')
        # 根据x生成查表的索引
        # 例如 L=2、start_pos=3 → [3,4]， 绝不能默认从 0 重新开始。
        position_ids = torch.arange(
            start_pos, start_pos + seq_len, device=x.device, dtype=torch.long
        )  # (L,)
        # 查表得到位置编码 (L, D)
        position_embeddings = self.pe(position_ids)
        # 广播相加：(B,L,D) + (L,D)
        return x + position_embeddings
