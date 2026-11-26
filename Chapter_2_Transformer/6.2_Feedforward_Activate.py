# 这个部分挺简单的，说白FFN就是先让同一个token内的特征交流一下呗，然后再进行类似于归一化的操作（保留重点信息）
# 至于什么是重要信息，那就是模型自己的学习结果了

class FFN(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()

        # 默认采用原始 Transformer 的 4 倍升维
        if d_ff is None:
            d_ff = 4 * d_model

        # 升维：学习不同的特征组合
        self.W1 = nn.Linear(d_model, d_ff)

        # 降维：重新组合中间特征
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, X):
        # X: (B, L, d_model)

        # 升维，学习中间特征
        H = self.W1(X)  # (B, L, d_ff)

        # 对中间特征进行非线性变换
        H = F.relu(H)  # (B, L, d_ff)

        # 降维，恢复原来的特征维度
        Y = self.W2(H)  # (B, L, d_model)

        return Y

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()

        # 默认采用 8/3 倍升维
        # 使权重矩阵的参数量与普通 FFN 大致相同
        if d_ff is None:
            d_ff = 8 * d_model // 3

        # 内容分支：学习需要传递的特征
        self.W_up = nn.Linear(d_model, d_ff, bias=False)

        # 门控分支：学习用于调节内容特征的信息
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)

        # 输出投影：恢复原来的特征维度
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, X):
        # X: (B, L, d_model)

        # 使用两个独立的权重矩阵学习两组特征
        A = self.W_up(X)    # (B, L, d_ff)
        G = self.W_gate(X)  # (B, L, d_ff)

        # 对门控特征进行非线性变换
        # SiLU(G) = G * sigmoid(G)
        gate = F.silu(G)  # (B, L, d_ff)

        # 逐元素乘法，利用门控特征调节内容特征
        H = A * gate  # (B, L, d_ff)

        # 重新组合中间特征，恢复原来的维度
        Y = self.W_down(H)  # (B, L, d_model)

        return Y
