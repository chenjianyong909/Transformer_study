
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Scaled Dot-Product Attention（缩放点积注意力）
# 回顾公式：Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + Mask) @ V

# 先实现注意力公式对应的计算过程，支持普通注意力和因果注意力
class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, is_causal=False, start_pos=0):
        # Q.shape = (B, L_q, D_k)
        # K.shape = (B, L_k, D_k)
        # V.shape = (B, L_k, D_v)
        # Q、K 的特征维度 D_k 必须相等，K、V 的 Token 数量 L_k 必须相等，但 L_q 不一定等于 L_k。

        # 1. 计算注意力分数，@ 表示矩阵乘法，transpose(-2,-1) 表示交换 K 的最后两个维度
        # K.T 会反转所有维度，这里存在 Batch 维度，因此使用 transpose(-2,-1)
        attention_scores = Q @ K.transpose(-2, -1)
        # (B, L_q, D_k) @ (B, D_k, L_k) -> (B, L_q, L_k)

        # 2. 缩放注意力分数，缓解特征维度增大导致的点积分数方差增大，避免 Softmax 过度饱和
        d_k = Q.size(-1)
        attention_scores = attention_scores / math.sqrt(d_k)
        # 使用 math.sqrt 而不是 torch.sqrt，因为 d_k 是整数，不是 Tensor

        # 3. Mask（可选），通过 is_causal 判断是否使用因果掩码
        if is_causal:
            # 调用 causal_mask() 生成布尔掩码矩阵，形状为 (L_q, L_k)
            mask = self.causal_mask(Q.size(1), K.size(1), start_pos=start_pos, device=Q.device)

            # masked_fill(condition, value)：将 condition 为 True 的位置替换成 value，其余位置保持不变
            # ~mask 表示布尔取反，因此将原本不允许关注的位置设置为 -inf，使其 Softmax 权重为 0
            attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        # 另一种实现方法：构建 Mask 时将允许关注的位置设为 0，需要遮挡的位置设为 -inf。然后直接执行 attention_scores + mask，不需要再使用 masked_fill()

        # 4. Softmax 归一化。attention_scores.shape = (B, L_q, L_k)，每一行表示一个 Query 对所有 Key 的匹配分数
        # 因此沿 L_k 维度归一化，即 dim=-1，使每个 Query 对所有 Key 的注意力权重之和为 1
        attn_weights = F.softmax(attention_scores, dim=-1)

        # 5. 根据注意力权重，对 V 进行加权求和
        output = attn_weights @ V # (B, L_q, L_k) @ (B, L_k, D_v) -> (B, L_q, D_v)

        return output

    # 构建 Causal Mask（因果掩码），不需要访问 self，因此使用静态方法
    @staticmethod
    def causal_mask(L_q, L_k, start_pos=0, device=None):
        # 为什么需要 Causal Mask？因为 GPT 在训练和生成时，当前 Token 不能关注未来 Token，避免未来信息泄漏。
        # 例如「苹果 很 甜」，假设每个词对应一个 Token。
        # token1「苹果」只能关注自身，token2「很」可以关注 token1、token2，token3「甜」可以关注全部三个 Token。

        # Mask 作用于 Softmax 前的注意力分数矩阵，attention_scores.shape = (B, L_q, L_k)
        # 行对应 Query，列对应 Key，因此 Mask.shape = (L_q, L_k)，可以广播到整个 Batch。

        # 情况一：完整序列的因果自注意力，L_q=L_k=3，start_pos=0
        # [[1, 0, 0],
        #  [1, 1, 0],
        #  [1, 1, 1]]

        # 完整序列的 Self-Attention 中，Q、K 来自同一个输入，因此 L_q=L_k。
        # Cross-Attention 的 Q、K 可以来自不同输入，因此 L_q、L_k 可以不同，但通常不使用因果掩码。

        # 情况二：KV Cache 增量推理，Q 可能只包含新增 Token，K 包含历史及新增 Token，因此 L_q 不一定等于 L_k。
        # start_pos 表示第一个 Query 在完整序列中的索引，假设 Key 从索引 0 开始。
        # 例如 L_q=2、L_k=5、start_pos=3，两个 Query 分别对应 token4、token5。
        # [[1, 1, 1, 1, 0],
        #  [1, 1, 1, 1, 1]]

        # 创建一个全为 False 的布尔矩阵，True 表示允许关注，False 表示遮挡
        mask = torch.zeros((L_q, L_k), dtype=torch.bool, device=device)

        # i 表示 Query 在当前 Q 中的索引，j 表示 Key 在 K 中的索引
        # start_pos+i 表示当前 Query 在完整序列中的索引
        for i in range(L_q):
            for j in range(L_k):
                if j <= i + start_pos:
                    mask[i, j] = True  # 只允许关注当前及之前的 Token

        # start_pos=0 表示第一个 Query 对应原序列索引 0，不一定代表完整序列。
        # start_pos>0 通常用于 KV Cache 增量推理，使新增 Query 能够正确对应历史 Key。
        # 例如已缓存「苹果、很」的 K，现在处理新增的「甜」，Q=[甜]，K=[苹果、很、甜]。
        # 此时 L_q=1、L_k=3，「甜」在 Q 中的索引 i=0，但在完整序列中的索引为 2，因此 start_pos=2。
        # 判断 j<=i+start_pos，即 j<=2，得到 Mask=[[1,1,1]]，表示「甜」可以关注全部三个 Token。
        # 如果错误地使用 start_pos=0，则得到 Mask=[[1,0,0]]，会错误遮挡「很」和「甜」。
        # 注意：逐 Token 解码时 K 中通常没有未来 Token，因此实际可以不使用显式因果掩码。

        return mask


# Self-Attention（自注意力机制）
# Q、K、V 都来自同一个输入 X，但分别经过不同的线性变换
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()

        # 初始化需要三个维度：
        # d_model：输入 X 中每个 Token 的特征维度
        # d_k：Q、K 经过线性投影后的特征维度，二者必须相等
        # d_v：V 经过线性投影后的特征维度，可以与 d_k 不同

        # 定义三个线性层，用于将输入 X 映射为 Q、K、V
        # nn.Linear 默认带有 bias，权重和偏置参数会在训练过程中更新
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)

        # 初始化缩放点积注意力模块，复用之前实现的注意力计算过程
        self.attention = DotProductAttention()

    def forward(self, X, is_causal=False):
        # X.shape = (B, L, d_model)
        # Q、K、V 都由同一个 X 生成，因此三者的 Token 数量相同，但特征维度可以不同

        # 1. 通过三个线性投影层生成 Q、K、V
        Q = self.W_Q(X)  # (B, L, d_model) -> (B, L, d_k)
        K = self.W_K(X)  # (B, L, d_model) -> (B, L, d_k)
        V = self.W_V(X)  # (B, L, d_model) -> (B, L, d_v)

        # 2. 调用缩放点积注意力，is_causal 决定是否使用因果掩码
        # 当前处理完整序列，因此 L_q=L_k=L，start_pos 使用默认值 0
        output = self.attention(Q, K, V, is_causal=is_causal)

        return output  # (B, L, d_v)



# Cross-Attention（交叉注意力机制）
# Q 来自输入 X，K、V 来自另一个输入 Y，两组输入可以具有不同的 Token 数量和特征维度
# 例如图像描述生成任务：Q 来自文本 Decoder，K、V 来自图像 Encoder，通过交叉注意力融合图像信息

class CrossAttention(nn.Module):
    def __init__(self, d_modelq, d_modelk, d_k, d_v):
        super().__init__()

        # 交叉注意力的 Q 和 K、V 来自不同输入，因此原始特征维度 d_modelq、d_modelk 可以不同
        # 经过线性投影后，Q、K 的特征维度必须相等，才能计算 Q @ K.T；V 的特征维度可以不同
        # d_modelq：输入 X 的特征维度；d_modelk：输入 Y 的特征维度
        # d_k：投影后的 Q、K 特征维度；d_v：投影后的 V 特征维度
        self.W_Q = nn.Linear(d_modelq, d_k)
        self.W_K = nn.Linear(d_modelk, d_k)
        self.W_V = nn.Linear(d_modelk, d_v)

        # 复用之前实现的缩放点积注意力
        self.attention = DotProductAttention()

    def forward(self, X, Y):
        # X.shape = (B, L_q, d_modelq)
        # Y.shape = (B, L_k, d_modelk)
        # X 用于生成 Q，Y 用于生成 K、V；L_q 和 L_k 可以不同

        Q = self.W_Q(X)  # (B, L_q, d_modelq) -> (B, L_q, d_k)
        K = self.W_K(Y)  # (B, L_k, d_modelk) -> (B, L_k, d_k)
        V = self.W_V(Y)  # (B, L_k, d_modelk) -> (B, L_k, d_v)

        # 调用缩放点积注意力，根据 Q 与 K 的匹配程度，对 V 进行加权求和
        # 普通 Cross-Attention 不需要因果掩码，默认 is_causal=False
        output = self.attention(Q, K, V)

        return output  # (B, L_q, d_v)



# 测试普通自注意力和因果自注意力
B, L, d_model = 2, 3, 8

X = torch.randn(B, L, d_model)
model = SelfAttention(d_model=8, d_k=4, d_v=6)

# 普通自注意力：可以关注完整序列
output1 = model(X, is_causal=False)

# 因果自注意力：只能关注当前及之前的 Token
output2 = model(X, is_causal=True)

print("输入 X：", X.shape)
print("普通自注意力输出：", output1.shape)
print("因果自注意力输出：", output2.shape)
print("因果掩码：")
print(DotProductAttention.causal_mask(L, L).int())

# 测试 Cross-Attention（交叉注意力）
# 模拟一个 Batch 中有 2 个样本，每个样本包含 X、Y 两组输入序列

B = 2           # Batch Size：每个批次包含 2 个样本
L_q = 3         # 每个样本的 X 中有 3 个 Token，用于生成 Query
L_k = 5         # 每个样本的 Y 中有 5 个 Token，用于生成 Key、Value
d_modelq = 8    # X 中每个 Token 的原始特征维度
d_modelk = 8    # Y 中每个 Token 的原始特征维度

# 构建两组输入序列，Batch 大小相同，但 Token 数量可以不同
# rand()是[0,1)均匀分布，randn()是标准正态分布，均值为0，方差为1
X = torch.randn(B, L_q, d_modelq)  # 按照标准正态分布生成数据，shape = (2, 3, 8)
Y = torch.randn(B, L_k, d_modelk)  # (2, 5, 8)

# 初始化交叉注意力模型
# 将 Q、K 投影到 4 维，将 V 投影到 6 维
cross_model = CrossAttention(d_modelq=8, d_modelk=8, d_k=4, d_v=6)

# 前向传播：Q 来自 X，K、V 来自 Y
cross_output = cross_model(X, Y)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("output.shape:", cross_output.shape)
