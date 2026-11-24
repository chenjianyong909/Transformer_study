# Rope旋转位置编码，其实很简单，我们只需要理解到一个核心点
# 为了给attention的点积运算加入位置信息，之前学习过的shaw、T5、transformer-xl、Alibi都是给输入x加位置信息，或者是计算后的点积加位置信息
# Rope就是利用点积的公式||q||*||k||cosa，去改变a的值从而实现加入相对位置信息。为什么这样可以加入相对位置信息？
# 这就需要考虑到cosa到底是以什么形式加入的了，去看我的ipynb文件，重点就在于那个旋转矩阵怎么出来的。

import math
import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    # 本质还是跟正余弦位置编码有关
    def __init__(self, d_model, num_heads, max_len=5000):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = max_len

        # 将 d_model 平均分配给每个注意力头。num_heads数量的头数分别组合成d_model
        self.d_head = d_model // num_heads

        # RoPE 在每个 Head 内部将特征维度两两分组
        # 因此需要保证 d_head 为偶数，而不只是 d_model 为偶数
        if self.d_head % 2 != 0:
            raise ValueError("d_head 必须为偶数")

        # 初始化 Q、K、V 的线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 多头注意力拼接后的输出投影
        self.W_o = nn.Linear(d_model, d_model)

        # 缩放系数只与每个 Head 的特征维度有关，可以提前计算
        self.scale = 1 / math.sqrt(self.d_head)

        # 提前计算并保存每个位置对应的正余弦旋转系数
        # RoPE 在每个 Head 内进行，因此传入 self.d_head，而不是整个 d_model
        self.SinAndCos(self.d_head, max_len)

    def SinAndCos(self, d_head, max_len):
        # 1. 每两个特征维度为一组，每组使用一个频率
        # 例如 d_head=4，分组为 (x0,x1)、(x2,x3)，t=[0,1]
        t = torch.arange(0, d_head // 2, dtype=torch.float32)
        theta = 1 / (10000 ** (2 * t / d_head))  # (d_head/2,)

        # 2. 每个 Token 的位置编号乘以每组的频率，得到旋转角度
        # pos[:, None] 是 (max_len,1)，theta[None,:] 是 (1,d_head/2)
        # angles[p,t] 表示位置 p 的 Token 在第 t 组上的旋转角度（弧度）
        pos = torch.arange(max_len, dtype=torch.float32)
        angles = pos[:, None] * theta[None, :]  # (max_len,d_head/2)

        # 3. 计算旋转矩阵需要的四个元素中的 sin 和 cos
        # 与原始正余弦位置编码不同，这里不将它们加到输入 x 上
        cos = torch.cos(angles)  # (max_len,d_head/2)
        sin = torch.sin(angles)  # (max_len,d_head/2)

        # 4. 显式构造每个位置、每个特征组对应的二维旋转矩阵
        # 对于角度 phi = angles[p,t]，逆时针旋转矩阵为：
        # R = [[cos(phi), -sin(phi)],   第一行
        #      [sin(phi),  cos(phi)]]   第二行
        # 先构造第一行和第二行，再将两行堆叠起来
        row1 = torch.stack([cos, -sin], dim=-1)  # (max_len,d_head/2,2)
        row2 = torch.stack([sin,  cos], dim=-1)  # (max_len,d_head/2,2)

        # dim=-2 在最后一个维度（列）之前增加行维度
        # R[p,t] 就是位置 p、第 t 组对应的完整 2×2 旋转矩阵
        R = torch.stack([row1, row2], dim=-2)  # (max_len,d_head/2,2,2)

        # R 是固定的，不参与梯度更新；随模型一起移动到 GPU
        # persistent=False 表示不写入 state_dict，因为可以重新计算
        self.register_buffer("R", R, persistent=False)

    def rotate(self, x, R):
        # x 是拆分多头之后的 Q 或 K：(B,H,L,d_head)
        # R 是对应位置、对应特征组的旋转矩阵：(L,d_head/2,2,2)
        B, H, L, d_head = x.shape

        # 1. 将一个 Token 的特征两两分组
        # 例如 [x0,x1,x2,x3] -> [[x0,x1],[x2,x3]]
        # 组数为 d_head/2，每组有两个特征
        x = x.reshape(B, H, L, d_head // 2, 2)  # (B,H,L,d_head/2,2)

        # 2. 将每组的两个特征写成列向量 [[a],[b]]
        # 这样才能按照 R @ x 的方式进行矩阵乘法：(2,2) @ (2,1)
        x = x.unsqueeze(-1)  # (B,H,L,d_head/2,2,1)

        # 3. 对每组特征执行旋转矩阵乘法
        # [[cos,-sin], [sin,cos]] @ [[a],[b]]
        # 得到 [[a*cos-b*sin], [a*sin+b*cos]]
        # @ 只对最后两个维度进行矩阵乘法
        # R 的形状 (L,d_head/2,2,2) 会自动广播到 Batch 和 Head 维度
        # 因此所有 Head 的相同位置、相同特征组使用相同的旋转矩阵
        x_rot = R @ x  # (B,H,L,d_head/2,2,1)

        # 4. 删除最后一个长度为 1 的维度，再把各组特征拼回原来的顺序
        # [[a0',b0'],[a1',b1']] -> [a0',b0',a1',b1']
        x_rot = x_rot.squeeze(-1)  # (B,H,L,d_head/2,2)
        x_rot = x_rot.reshape(B, H, L, d_head)  # (B,H,L,d_head)
        return x_rot

    def forward(self, x, is_causal=False):
        # 双向或因果 Self-Attention，Q、K、V 来自同一个输入 x
        # x: (B,L,d_model)，当前实现的位置编号从 0 开始，不包含 KV Cache
        B, L, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError("输入特征维度与 d_model 不一致")
        if L == 0 or L > self.max_len:
            raise ValueError("输入序列长度不合法")

        # 1. 线性投影得到 Q、K、V，不需要提前在 x 上加位置向量
        Q = self.W_q(x)  # (B,L,d_model)
        K = self.W_k(x)  # (B,L,d_model)
        V = self.W_v(x)  # (B,L,d_model)

        # 2. 拆分多头：(B,L,d_model) -> (B,L,H,d_head) -> (B,H,L,d_head)
        Q = Q.reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)
        K = K.reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)

        # 3. 取出当前序列需要的旋转矩阵，Q 和 K 共用同一张表
        # 例如 L=3，只取位置 0、1、2 的旋转矩阵
        R = self.R[:L].to(Q.dtype)  # (L,d_head/2,2,2)

        # 4. 分别按照自身绝对位置旋转 Q、K，不旋转 V
        # 每组中 (R_i @ q_i).T @ (R_j @ k_j) = q_i.T @ (R_i.T @ R_j) @ k_j
        # R_i.T @ R_j 等价于旋转 (j-i)*theta，因此点积中的位置变换只依赖相对距离
        Q = self.rotate(Q, R)  # (B,H,L,d_head)
        K = self.rotate(K, R)  # (B,H,L,d_head)

        # 5. 旋转之后，仍然使用普通 Attention 的缩放点积
        scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B,H,L,L)

        # 6. 可选因果 Mask：行对应 Query，列对应 Key，只允许 j <= i
        if is_causal:
            pos = torch.arange(L, device=x.device)
            mask = pos[None, :] <= pos[:, None]  # (L,L)
            scores = scores.masked_fill(~mask, float("-inf"))

        # 7. 沿 Key 维度归一化，再对 V 加权求和
        weights = torch.softmax(scores, dim=-1)  # (B,H,L,L)
        output = weights @ V  # (B,H,L,d_head)

        # 8. 拼接多个 Head，再进行输出投影
        output = output.transpose(1, 2).reshape(B, L, self.d_model)
        output = self.W_o(output)  # (B,L,d_model)
        return output

if __name__ == "__main__":
    torch.manual_seed(0)
    model = RotaryPositionEmbedding(d_model=8, num_heads=2, max_len=10)
    x = torch.randn(2, 3, 8)
    print("旋转矩阵表形状：", model.R.shape)
    print("位置 1、第 0 组的旋转矩阵：\n", model.R[1, 0])
    print("双向注意力输出形状：", model(x).shape)
    print("因果注意力输出形状：", model(x, is_causal=True).shape)
    
        