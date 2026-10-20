import torch
import torch.nn as nn
import math

# 复现一下正余弦位置编码
class SinusoidalPositionalEncoding(nn.Module):
    # 首先初始化函数得搞明白需要传入什么参数
    # 正余弦最重要的就是位置编码矩阵PE的计算，如何理解呢？
    # PE最终就是要跟输入的X进行相加，所以从X的维度去理解就可以了。
    # X的维度是(B, L, D)，其中B是batch size，L是序列长度，D是词向量维度。
    # 所以PE的维度也应该是(B, L, D)，但是每一个batch可以使用相同的PE。
    # 位置记录的是单条句子内部token的先后顺序，相同位置序号在不同句子里含义一样，所以batch里所有样本可以共用一套PE。
    # 最终的PE只要（L,D）就可以了
    def __init__(self, d_model: int, max_len:int):
        super().__init__()
        # 初始化需要做的事情就是把PE给算出来，这样子调用SinusoidalPositionalEncoding对象的时候自动执行前向传播，直接完成位置编码
        # 依旧是回到位置编码的公式，sin/cos的角度计算公式是：
        # sin(pos, 2i) = pos / 10000^(2i/dmodel)
        # cos(pos, 2i+1) = pos / 10000^(2i/dmodel)
        # 其中pos是位置序号，i是维度序号，dmodel是词向量维度。(需要注意的是，这里面全部都是2i)

        # 接下来就是复现公式了
        # 1.确定pos：先把每一个token的初始位置确定一下 [token1, token2, token3, ...] -> [0, 1, 2, ...]
        positions = torch.arange(0, max_len, dtype=torch.float32) # (max_len,)
        # 2.确定角速度，就是正余弦函数变化的频率。sin/cos共同构成1组，每组的输入是一样的，所以只需要一半的d_model就可以了
        i = torch.arange(0, d_model//2, dtype=torch.float32) # (d_model//2,)
        freq = torch.exp(-math.log(10000) * (2*i/d_model)) # (d_model//2, )

        # 计算传入sin/cos的输入
        # angles = positions * freq # (max_len,) * (d_model/2,)。对应位置相乘，维度不对，会报错的
        # 所以需要把positions和freq变成可以计算的样子 
        positions = positions.unsqueeze(1) # (max_len, 1)
        freq = freq.unsqueeze(0) # (1, d_model/2)
        # 这样子计算的时候就可以都广播成(max_len, d_model/2)进行计算了
        angles = positions * freq # (max_len, 1) * (1, d_model/2) -> (max_len, d_model/2)

        # 计算sin和cos值
        sin = torch.sin(angles) # (max_len, d_model/2)
        cos = torch.cos(angles) # (max_len, d_model/2)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32) # (max_len, d_model)
        pe[:, 0::2] = sin # 奇数做sin
        pe[:, 1::2] = cos # 偶数做cos

        # 将PE注册为buffer，这样在模型保存和加载时也会被保存（推理时能用）
        # 什么时候用register_buffer：张量是模型预先算好的常量，
        # 不想每次前向推理重复计算，需要跟着模型保存加载、自动切换CPU/GPU，但是不需要训练更新。
        # 以PE举例：PE在__init__一次性算完。如果不注册buffer，保存模型时PE不会被存下来，
        # 下次加载模型就丢失PE，必须重新计算；同时model.to()不会自动移动普通self.pe到GPU。
        # 不能用Parameter：会计算梯度，训练会篡改固定PE的值。
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # 这个就很简单了，就是你调用对象的时候想做什么事情，就在这里面写就可以了
        # 我们想做的事情就是把x跟PE相加，引入位置信息。所以输入参数得有x
        # 不传PE的原因是PE已经在__init__里面算好了，直接用self.pe就可以了

        # x的维度是(B, L, D)，L是本次输入实际token个数
        # self.pe是(max_len, D)，max_len是初始化预设的最大序列长度，没有额外batch维度
        # 训练阶段会把batch内样本padding补齐到全局max_len，直接相加没问题；
        # 推理一般不padding；批量推理时可能会padding到当前batch的最长长度，但不会统一补齐到全局max_len，输入长度仍然可变；
        # 自回归续写时还会每次只输入1个token，所以必须根据start_pos截取对应PE片段
        # start_pos代表当前这批token在完整序列里的起始位置，支持prompt输入和增量生成

        # 安全检测一下吧，免得超过了
        if start_pos < 0 or start_pos + x.size(1) > self.pe.size(0):
            raise ValueError('start_pos 或序列长度超出了 max_len。')

        L = x.size(1) # 本次输入的token个数
        # 从pe中截取出本次输入对应的PE片段，注意要加上start_pos
        pe_slice = self.pe[start_pos:start_pos+L, :] # (max_len, D) -> (L, D)

        return x + pe_slice

if __name__ == '__main__':
    # 测试一下
    position_encoding = SinusoidalPositionalEncoding(d_model=8, max_len=128)
    print('可训练参数数量:', sum(p.numel() for p in position_encoding.parameters()))
    print('已注册的 buffer:', list(dict(position_encoding.named_buffers()).keys()))
    print('位置矩阵形状:', tuple(position_encoding.pe.shape))
