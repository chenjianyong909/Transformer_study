import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    def __init__(self, F):
        super().__init__()
        # F 是特征变换模块，例如 Attention 或 FFN
        self.F = F

    def forward(self, x):
        # x: (B, L, d_model)，要求 F(x) 与 x 形状一致
        # self.F(x)：调用传入的特征变换模块；若 F 是 Linear，则为 x @ W.T + b
        # 直接加上原始输入，为梯度提供恒等传播路径
        return x + self.F(x)  # (B, L, d_model)


class PostNorm(nn.Module):
    def __init__(self, F, norm):
        super().__init__()
        self.F = F
        # norm 可以是 LayerNorm 或 RMSNorm
        self.norm = norm

    def forward(self, x):
        # 先进行特征变换，再与原始输入相加
        y = x + self.F(x)  # (B, L, d_model)

        # 对相加后的结果归一化，直接控制输出尺度
        # 此时残差路径也需要经过归一化层
        y = self.norm(y)  # (B, L, d_model)
        return y


class PreNorm(nn.Module):
    def __init__(self, F, norm):
        super().__init__()
        self.F = F
        self.norm = norm

    def forward(self, x):
        # 先归一化，控制特征变换模块的输入尺度
        h = self.norm(x)  # (B, L, d_model)
        h = self.F(h)     # (B, L, d_model)

        # 加的是原始 x，而不是归一化后的 x
        # 残差路径不经过归一化，保留恒等梯度路径
        y = x + h  # (B, L, d_model)
        return y


if __name__ == "__main__":
    d_model = 4
    x = torch.randn(2, 3, d_model)  # (B, L, d_model)

    # 用普通线性层代替 Attention / FFN，只演示连接方式。
    # nn.Linear(d_model, d_model) 内部封装了可学习的权重 W 和偏置 b：
    # W: (out_features, in_features) = (d_model, d_model)，b: (d_model,)
    # 创建线性层时只初始化参数，并未计算；调用 F(x) 时才执行 x @ W.T + b。
    # 输入和输出特征维度相同，才能与原始 x 进行残差相加。
    # nn.LayerNorm(d_model) 则创建一个对最后一维归一化的模块。

    # 将线性层对象传给 F，计算 y = x + F(x)
    residual = ResidualConnection(nn.Linear(d_model, d_model))

    # 传入线性层 F 和归一化层 norm，计算 y = norm(x + F(x))
    post_norm = PostNorm(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    # 传入线性层 F 和归一化层 norm，计算 y = x + F(norm(x))
    pre_norm = PreNorm(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    # 上面仅创建并保存网络模块，下面传入 x 才执行各自的 forward()。

    print("Input:   ", x.shape)
    print("Residual:", residual(x).shape)
    print("PostNorm:", post_norm(x).shape)
    print("PreNorm: ", pre_norm(x).shape)
