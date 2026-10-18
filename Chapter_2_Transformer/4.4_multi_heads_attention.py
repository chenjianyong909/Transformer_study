import torch
import torch.nn as nn
import math


# Multi-Head Attention（多头注意力）
# 将 Q、K、V 分成多个头，每个头独立计算注意力

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        # d_model: 输入的特征维度
        # num_heads: 注意力头的数量
        self.num_heads = num_heads

        # 这里直接采用 Transformer 原论文中的常见做法，
        # 将 d_model 平均分配给每个注意力头
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("num_heads 必须为正数，且 d_model 必须能被 num_heads 整除")

        # 这边就需要自己计算d_k 和 d_v了，因为得根据头的数量来计算
        self.d_k = d_model // num_heads  # 每个 Head 的 Q、K 特征维度
        self.d_v = d_model // num_heads  # 每个 Head 的 V 特征维度

        # 我们要一次性计算所有注意力头，
        # 因此需要生成 num_heads 组 Q、K、V。
        #
        # 例如 d_model=8, num_heads=2：每个头的维度为 4，总输出维度为 2*4=8
        self.W_Q = nn.Linear(d_model, self.d_k * num_heads)
        self.W_K = nn.Linear(d_model, self.d_k * num_heads)
        self.W_V = nn.Linear(d_model, self.d_v * num_heads)

        # W_O：多头拼接后的线性投影
        # 作用是让模型学习如何组合不同注意力头的信息
        self.W_O = nn.Linear(self.d_v * num_heads, d_model)


    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力（Scaled Dot-Product Attention）

        单头注意力：
        Q.shape = (B, L_q, d_k)
        K.shape = (B, L_k, d_k)
        V.shape = (B, L_k, d_v)

        多头注意力，把头作为维度，便于遍历：
        Q.shape = (B, H, L_q, d_k)
        K.shape = (B, H, L_k, d_k)
        V.shape = (B, H, L_k, d_v)

        注意：
        多头相比单头只是多了 H 这个维度。
        torch.matmul 会把 B、H 作为批量维度，
        分别计算每个样本中每个头的注意力。
        """

        # ==========================================
        # 1. 计算 Q @ K.T
        # ==========================================
        #
        # B、H 作为批量维度，不参与矩阵乘法中的点积求和。 matmul只把最后两维拿来做矩阵乘法。前面的都当作批量维度。
        # scores.shape = (B, H, L_q, L_k)

        scores = torch.matmul( Q, K.transpose(-2, -1) ) / math.sqrt(self.d_k)

        # ==========================================
        # 2. 添加注意力掩码（可选）
        # ==========================================
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))

        # ==========================================
        # 3. Softmax 计算注意力权重
        # ==========================================
        # attention_weights.shape = (B, H, L_q, L_k)
        # 注意力机制的目的是：对于每一个 Query，计算它应该关注哪些 Key，以及分别关注多少。
        # 所以是针对L_k的维度归一化，注意力算完刚刚好L_q, L_k，表示query 对应 key的分数
        attention_weights = torch.softmax(scores, dim=-1)

        # ==========================================
        # 4. 注意力权重与 V 相乘
        # ==========================================

        # attention_weights.shape = (B, H, L_q, L_k)
        # V.shape                 = (B, H, L_k, d_v)
        #
        # 矩阵乘法：(L_q, L_k) @ (L_k, d_v) -> (L_q, d_v)
        # output.shape = (B, H, L_q, d_v)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


    def forward(self, x, mask=None):
        # 前向传播，现在传进来的 x 是一个三维张量，形状为 (B, L, d_model)，这个是不变的
        B, L, _ = x.shape
        H = self.num_heads

        # 当前实现的是自注意力，所以 L_q = L_k = L
        L_q = L

        # 1. 进行线性变换，得到 Q、K、V
        # 一次性生成所有注意力头的 Q、K、V
        Q = self.W_Q(x)  # (B, L, H*d_k)
        K = self.W_K(x)  # (B, L, H*d_k)
        V = self.W_V(x)  # (B, L, H*d_v)

        # 2. 将 Q、K、V reshape 成多头的形式
        # reshape：将特征维度拆成 H 个头
        # (B, L, H*d_k) -> (B, L, H, d_k)
        #
        # transpose(1, 2)：交换 L 和 H 的位置
        # (B, L, H, d_k) -> (B, H, L, d_k)
        #
        # 因为 matmul 默认使用最后两个维度执行矩阵乘法，所以将 B、H 放到前面作为批量维度。

        Q = Q.reshape(B, L, H, self.d_k).transpose(1, 2)
        K = K.reshape(B, L, H, self.d_k).transpose(1, 2)
        V = V.reshape(B, L, H, self.d_v).transpose(1, 2)

        # Q.shape = (B, H, L, d_k)
        # K.shape = (B, H, L, d_k)
        # V.shape = (B, H, L, d_v)

        # 3. 调用缩放点积注意力函数
        # 每个样本的每个 Head 都独立计算注意力
        output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )

        # output.shape = (B, H, L_q, d_v)
        # attention_weights.shape = (B, H, L_q, L)
        # 4. 拼接多个注意力头（Concat）。需要将同一个 token 在不同 Head 上的输出拼接起来。
        # 先交换 H 和 L_q：(B, H, L_q, d_v) -> (B, L_q, H, d_v)
        output = output.transpose(1, 2)

        # 再合并 H 和 d_v：(B, L_q, H, d_v) -> (B, L_q, H*d_v)
        # contiguous()：必要时整理为连续内存布局
        # view()：合并最后两个维度，实现 Concat
        output = output.contiguous().view(B, L_q, H * self.d_v)

        # 5. 最后通过 W_O 进行线性投影,将多个 Head 拼接后的特征映射回 d_model 维度
        output = self.W_O(output)  # (B, L_q, d_model)

        return output, attention_weights
