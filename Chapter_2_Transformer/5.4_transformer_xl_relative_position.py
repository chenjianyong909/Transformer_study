'''
代码书写transformer-xl相对位置编码，与之前相同，围绕公式去进行：
  Transformer-XL：q_i·k_j + q_i·r_ij + u·k_j + v·r_ij，
    其中 r_ij 来自 Sinusoidal 相对距离编码的可学习投影。
'''

import torch
import torch.nn as nn
import math

# 直接看class，不用从头看
def SinusoidalRelativePositionalEncoding(D, d_model):
    # 正余弦相对位置编码需要输入编码维度
    # 位置索引,按组来分索引,因为是d_model进行计算，所以肯定是给d_model加位置信息
    index = torch.arange(0, d_model, 2).float() # (d_model/2,)

    # 频率值
    freq = 1 / (10000 ** (index / d_model)) # (d_model/2,)

    # 位置编码矩阵，因为我们是根据相对位置矩阵进行的编码
    # D 表示的就是相对位置矩阵 (L_q, L_k)，就是根据相对位置的值去计算对应的位置编码
    # 正余弦的输入angle由 D 和 freq 计算得到，进行运算就需要考虑维度变换
    # (L_q, L_k)无法直接跟(d_model/2,)进行计算，并且还需要考虑到freq的每一个值都需要跟D的每一个值进行计算
    # 所以维度变换也是有讲究的
    D = D.unsqueeze(-1).float() # (L_q, L_k, 1)
    freq = freq.unsqueeze(0).unsqueeze(0) # (1, 1, d_model/2)
    angle = D * freq # (L_q, L_k, d_model/2)

    # angle 表示每一组的正余弦输入，组表示的是什么呢？ 比如d_model的第0，1个维度共用一个angle， 正余弦算完的值就是sin、cos、sin、cos....
    sin = torch.sin(angle) # (L_q, L_k, d_model/2)
    cos = torch.cos(angle) # (L_q, L_k, d_model/2)
    # 需要交叉拼在一起，得到 [sin0, cos0, sin1, cos1] 相当于 (L_q, L_k, d_model/2) -> (L_q, L_k, d_model)
    # 这里需要引入一个函数 stack，来进行交叉拼接
    # 对于一维向量，dim=0 表示按行排列，dim=-1 表示按列排列
    # [sin0,sin1] 和 [cos0,cos1] 按列排列后变为：
    # [[sin0,cos0],
    #  [sin1,cos1]]
    # 对于这里的三维张量，dim=-1 表示在最后增加一个维度
    pos_enc = torch.stack((sin, cos), dim=-1)  
    # 接下来还需要把 [[sin0, sin1], [cos0, cos1]] 变为 [sin0, cos0, sin1, cos1]，也就是使用flatten()函数，只按行拼接
    # 这里会发现[[sin0, sin1], [cos0, cos1]]按行拼接以后变为 [sin0, sin1, cos0, cos1]，所以上面的stack函数需要是dim=-1
    # 所以需要让 sin 和 cos 按列排列，得到 [[sin0,cos0],[sin1,cos1]]
    # 因此上面的 stack 使用 dim=-1，再通过 flatten(-2) 按行展开，这样拼接以后就变为 [sin0, cos0, sin1, cos1]了
    pos_enc = pos_enc.flatten(-2) # -2表示只拼接最后两维，前面的不变。(L_q, L_k, d_model/2, 2) -> (L_q, L_k, d_model)

    return pos_enc

class RelativePositionEmbedding(nn.Module):
    def __init__(self, d_model):
        # 在创建该对象时需要输入哪些参数如何进行判断？ 
        # 可以初步根据我们需要的东西判断，也可以后续编写代码过程中添加， 我更倾向于后一种

        super().__init__()

        # 相对位置编码也是用于注意力部分的，这里我们讨论的Self-attention
        # 所以常规来说需要进行W_Q,W_K,W_V这三个权重矩阵的初始化
        # 这里就出现了需要输入d_model的情况，所以函数的参数中有d_model
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # 前向传播是直接使用进行计算，所以我们需要把计算要用的都是准备好了
        # Transformer-XL：q_i·k_j + q_i·r_ij + u·k_j + v·r_ij
        # 除了数据以外的都需要准备，也就是r_ij、u、v
        # 先看 r_ij，它并不是直接学习出来的一个固定向量
        # 而是根据 Query 和 Key 的相对距离，先计算 Sinusoidal 位置编码 R(i-j)
        # 再通过可学习矩阵 W_R 进行线性投影，得到 r_ij
        # r_ij = W_R * R(i-j)
        # 因此 W_R 需要接收位置编码作为输入，而且其计算的对象是矩阵，所以使用 nn.Linear
        self.W_R = nn.Linear(d_model, d_model, bias=False)

        # 为什么 u、v 不根据每个 Query 分别生成？
        # 因为公式中的 q_i 已经负责表达每个 Query 自身的内容
        # u、v 则用于学习所有 Query 共享的内容偏好和相对位置偏好
        # 因此，在单头 Attention 中，所有 Query 共用一组 u、v
        # 注意：共享参数不代表注意力分数相同，因为 k_j 和 r_ij 仍然会变化
        # 如果扩展为多头 Attention，则每个 Head 会分别学习自己的 u、v
        # 共享的话那u、v只需要是向量就可以了
        self.u = nn.Parameter(torch.zeros(d_model)) # (d_model,)
        self.v = nn.Parameter(torch.zeros(d_model)) # (d_model,)



    def forward(self, X):
        # 这里先实现不带历史记忆的单头因果 Self-Attention
        # X：(B,L,d_model)
        B, L, d_model = X.shape

        # 1. 计算 Q、K、V
        # Self-Attention 中，Q、K、V 都来自同一个输入 X
        Q = self.W_Q(X)  # (B,L,d_model)
        K = self.W_K(X)  # (B,L,d_model)
        V = self.W_V(X)  # (B,L,d_model)

        # 2. 计算相对距离矩阵
        # 行对应 Query，列对应 Key，D[i,j]=i-j
        pos = torch.arange(L, device=X.device)
        D = pos[:, None] - pos[None, :]  # (L,L)

        # 3. 计算相对位置向量
        # 每个 Query-Key 对应一个相对距离，也就对应一个位置向量
        # Sinusoidal 编码后，再通过 W_R 进行可学习投影
        pos_enc = SinusoidalRelativePositionalEncoding(D, d_model)
        R = self.W_R(pos_enc)  # (L,L,d_model)

        # 4. 第一项：q_i · k_j
        # 每个 Query 都需要和所有 Key 计算点积
        # Q：(B,L,d_model)
        # K.transpose(-2,-1)：(B,d_model,L)
        # 矩阵相乘后，每一行对应一个 Query，每一列对应一个 Key
        score_1 = Q @ K.transpose(-2, -1)  # (B,L,L)

        # 5. 第二项：q_i · r_ij
        # 这里不能直接使用 Q @ R，因为 R 不是普通的 (L,d_model)
        # R 的形状是 (L,L,d_model)，每对 Query-Key 都有自己的位置向量

        # Q[b,i,:] 表示第 b 个 Batch 中，第 i 个 Query 的向量
        # R[i,j,:] 表示第 i 个 Query 与第 j 个 Key 的相对位置向量
        # 我们希望计算：Q[b,i,:] 与 R[i,j,:] 的点积

        # Q 原来是 (B,L,d_model)，缺少 Key 这一维
        # 所以在 Query 维度后面插入一个维度：
        # (B,L,d_model) -> (B,L,1,d_model)
        # 这个 1 代表暂时只有一个位置，广播时会扩展到所有 Key
        # 这样同一个 Query 就能与所有 Key 对应的位置向量计算点积
        Q_expand = Q.unsqueeze(2)  # (B,L,1,d_model)

        # R 原来是 (L,L,d_model)，缺少 Batch 这一维
        # 因为相对距离只取决于 Query 和 Key 的位置，不取决于 Batch
        # 所以需要让不同 Batch 共用同一个相对位置向量矩阵
        # 在最前面插入一个维度：
        # (L,L,d_model) -> (1,L,L,d_model)
        R_expand = R.unsqueeze(0)  # (1,L,L,d_model)

        # 现在两个张量可以广播相乘：
        # Q_expand：(B,L,1,d_model)
        # R_expand：(1,L,L,d_model)
        #             ↓  ↓ ↓   ↓
        # 广播结果： (B,L,L,d_model)
        #
        # Batch：R 的 1 扩展成 B，让每个 Batch 使用相同的 R
        # Query：两边都是 L，表示第 i 行对应第 i 个 Query
        # Key：Q 的 1 扩展成 L，让同一个 Query 与所有 Key 的位置向量相乘
        # 特征：两边都是 d_model，表示两个向量逐元素相乘

        # 逐元素相乘之后，还需要沿特征维度求和，才是向量点积
        # (B,L,L,d_model) -> (B,L,L)
        score_2 = (Q_expand * R_expand).sum(dim=-1)

        # 6. 第三项：u · k_j
        # u 是所有 Query 共享的一个可学习向量，形状为 (d_model,)
        # 它需要与每一个 Key 计算点积，但是不需要根据 Query 分别计算

        # K：(B,L,d_model)
        # u：(d_model,)
        # 广播时，u 会与每个 Batch 中的每个 Key 相乘
        # 相当于 K[b,j,:] 与 u 计算点积
        # 沿特征维度求和之后，得到 (B,L)
        score_3 = (K * self.u).sum(dim=-1)  # (B,L)

        # 但是其他分数是 (B,L,L)，分别有 Query 和 Key 两个位置维度
        # score_3 只有 Key 维度，没有 Query 维度
        # 因为 u 是共享的，所以对于同一个 Key，不同 Query 使用相同的这一项分数
        # 因此在 Key 维度前面增加一个维度：
        # (B,L) -> (B,1,L)
        # 后面相加时，这个 1 会广播成 L，即所有 Query 共用该 Key 的分数
        score_3 = score_3.unsqueeze(1)  # (B,1,L)

        # 7. 第四项：v · r_ij
        # v 也是所有 Query 共享的可学习向量，形状为 (d_model,)
        # 但 R[i,j,:] 会随着 Query 和 Key 的相对距离发生变化

        # R：(L,L,d_model)
        # v：(d_model,)
        # 广播时，每个位置向量 R[i,j,:] 都与同一个 v 逐元素相乘
        # 沿特征维度求和，得到每对 Query-Key 的位置分数
        # (L,L,d_model) -> (L,L)
        score_4 = (R * self.v).sum(dim=-1)  # (L,L)

        # score_4 已经有 Query 和 Key 两个维度，但是没有 Batch 维度
        # 因为不同 Batch 共用相同的 v 和相对距离位置编码
        # 所以在最前面增加一个维度：
        # (L,L) -> (1,L,L)
        # 后面相加时，这个 1 会广播成 B
        score_4 = score_4.unsqueeze(0)  # (1,L,L)

        # 8. 四项分数相加
        # score_1：(B,L,L)
        # score_2：(B,L,L)
        # score_3：(B,1,L) -> 广播成 (B,L,L)
        # score_4：(1,L,L) -> 广播成 (B,L,L)
        # 四项在同一个 Batch、Query、Key 位置上相加
        scores = (score_1 + score_2 + score_3 + score_4) / math.sqrt(d_model)

        # 9. 因果 Mask
        # D[i,j]=i-j，当 D>=0 时，表示 Key 不在 Query 的未来
        # 因此允许关注；否则需要屏蔽
        causal_mask = D >= 0  # (L,L)

        # scores 有 Batch 维，而 causal_mask 没有
        # 因此增加 Batch 维：(L,L) -> (1,L,L)
        # 广播后，每个 Batch 使用相同的因果 Mask
        # ~ 表示取反，把不允许关注的位置设为 -inf
        scores = scores.masked_fill(
            ~causal_mask.unsqueeze(0),
            float("-inf")
        )

        # 10. Softmax
        # 每一行是一个 Query 对所有 Key 的分数
        # dim=-1 表示沿 Key 维度归一化
        attn = torch.softmax(scores, dim=-1)  # (B,L,L)

        # 11. 与 V 相乘，得到 Attention 输出
        # 每个 Query 根据自己的注意力权重，对所有 Value 加权求和
        output = attn @ V  # (B,L,d_model)

        return output

if __name__ == "__main__":
    # 测试
    d_model = 8
    B, L = 2, 4
    X = torch.randn(B, L, d_model)
    model = RelativePositionEmbedding(d_model)
    output = model(X)
    print(output.shape)  # (B,L,d_model)
