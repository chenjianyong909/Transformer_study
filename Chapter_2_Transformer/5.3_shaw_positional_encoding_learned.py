
import torch
import torch.nn as nn
import math

class ShawRelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_distance = 2):
        # 如何确认输入的参数是哪些？-> 明白相对可学习位置编码的原理是什么，需要什么？
        # 需要两个相对位置向量表，分别用于K和V，大小为[2*max_distance+1, d_model]
        # 由于为了防止参数太多了，所以还需要设置一个max_distance，超过这个距离的相对位置编码就不再区分了，直接用同一个编码表示。
        # Q,K,V矩阵的生成需要使用d_model
        super().__init__()
        self.d_model = d_model
        self.r = max_distance

        # 初始化一下相对位置矩阵，K需要一个，V也需要一个。因为是在转化为K,V后引入的，所以各一个，根据实际情况也可以共用一个
        # 由于是可学习的位置向量，所以直接使nn.Embedding就可以了
        self.E_K = nn.Embedding(2 * self.r + 1, d_model)
        self.E_V = nn.Embedding(2 * self.r + 1, d_model)
        # 设定了最大的相对距离 r， 相对位置矩阵的行数就是相对位置的个数，列就是各自的d_model
        # 所以相对位置可以是 [-r,r]，一共 2r+1 个相对位置。Embedding 的输入就是相对位置的索引，输出就是对应的向量表示。

        # 进行 W_Q, W_K, W_V 参数矩阵的初始化
        # Q、K、V 都由同一个输入 X 经过不同的线性投影得到
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        # 前向传播就要注意矩阵乘法规则跟广播规则就好写了
        # 主要就是写这些向量之间是怎么运算的

        # 1. 先计算 Q、K、V
        # X.shape = (B, L, d_model)
        Q = self.W_q(X)  # (B, L, d_model)
        K = self.W_k(X)  # (B, L, d_model)
        V = self.W_v(X)  # (B, L, d_model)

        # 2. 计算相对位置矩阵
        # 先计算相对位置矩阵 D，D[i,j] = j - i
        # 这里的 D 是一个二维矩阵，大小为 (L, L)，表示每个位置之间的相对位置关系
        L = X.size(1)
        position = torch.arange(L, device=X.device)  # (L,)
        # 这里就可以利用广播复制元素值的特性，让它变成一个(L, L)的矩阵

        # print('position:', position)
        # print('position[:, None]:', position.unsqueeze(1)) # (L, 1)
        # print('position[None, :]:', position.unsqueeze(0)) # (1, L)
        print(position.unsqueeze(1) - position)
        print(position - position.unsqueeze(1))
        # Key 位置减去 Query 位置
        D = position.unsqueeze(0) - position.unsqueeze(1)  # (1, L) - (L, 1) -> (L, L)，每个位置之间的相对位置关系
        # Key 的位置 [0,1,2] 减去第一个 Query 的位置 0，得到相对距离 [0,1,2]，表示第一个 Query 与各个 Key 的相对位置关系。
        # 记住这个就知道是谁减去谁了

        # 将相对位置矩阵 D 的数值裁剪到 [-r, r] 范围内，但不改变矩阵形状，r=1时:[-2,-1,0,1,2] -> [-1,-1,0,1,1]，超出范围的距离使用边界对应的位置向量。
        D.clamp_(-self.r, self.r)
        # 将相对位置矩阵 D 映射到 [0, 2r] 范围内，作为 Embedding 的索引，便于查表，因为查表是从 0 开始的。
        D_index = D + self.r  # (L, L)，索引范围 [0, 2r]

        # 通过 Embedding 查表得到相对位置向量
        # 这里的 E_K 和 E_V 是两个 Embedding 层，分别用于查找相对位置向量，大小为 (2r+1, d_model)
        A_K = self.E_K(D_index)  # (L, L, d_model)
        A_V = self.E_V(D_index)  # (L, L, d_model)

        # 加入相对位置以后开始计算注意力 'Q @ (K + A_K).T'，每一次进行矩阵计算的时候都要注意维度
        # Q.shape = (B, L, d_model)；K.shape = (B, L, d_model)；A_K.shape = (L, L, d_model)
        # 我们要进行计算的部分是每一个batch的(L, L, d_model)，所以需要进行维度的扩展
        # K: (B, L, d_model)，L 对应 Key 的位置
        # A_K: (L, L, d_model)，第一个 L 是行（Query），第二个 L 是列（Key）
        # 在 K 的 Query 维度增加一个大小为 1 的维度
        # (B, L, d_model) -> (B, 1, L, d_model)
        # 广播时会沿行（Query）方向复制，让每个 Query 都对应完整的一组 Key
        K_relative = K.unsqueeze(1) # (B, 1, L, d_model)，所以这里不能写 K.unsqueeze(2)，会变成按列复制，[k0,k0,k0,k0]
        # A_K 没有 Batch 维度，因此在最前面增加一个大小为 1 的维度
        # 广播时所有 Batch 共用同一张相对位置向量表
        A_K_relative = A_K.unsqueeze(0) # (1, L, L, d_model)
        K_plus_A_K = K_relative + A_K_relative # (B, L, L, d_model)
        # Q 要进行内积计算的话也需要构造出维度，然后使用torch.matmul进行矩阵乘法，只计算后两维
        Q_relative = Q.unsqueeze(2) # (B, L, 1, d_model)
        # 计算注意力分数
        attention_scores = torch.matmul(Q_relative, K_plus_A_K.transpose(-2, -1)) / math.sqrt(self.d_model) # (B, L, 1, L)

        # 对每个 Query 的 L 个 Key 分数进行 Softmax，得到注意力权重
        attenttion_weigths = torch.softmax(attention_scores, dim=-1) # (B, L, 1, L)

        # 接下来就是跟 V 进行注意力加权
        V_relative = V.unsqueeze(1) # (B, 1, L, d_model)
        A_V_relative = A_V.unsqueeze(0) # (1, L, L, d_model)
        V_plus_A_V = V_relative + A_V_relative # (B, L, L, d_model)
        # 使用 Softmax 后的注意力权重计算输出
        attention_output = torch.matmul(attenttion_weigths, V_plus_A_V) # (B, L, 1, d_model)

        # 最终的输出应该是跟X一样的，所以需要去掉中间的维度
        output = attention_output.squeeze(2) # (B, L, d_model)
        return output, attenttion_weigths



if __name__ == '__main__':
    # 测试一下 ShawRelativePositionalEncoding 的前向传播
    d_model = 4
    max_distance = 2
    model = ShawRelativePositionalEncoding(d_model, max_distance)
    X = torch.randn(2, 5, d_model)  # (B=2, L=5, d_model=4)
    output, attenttion_weigths = model(X)
