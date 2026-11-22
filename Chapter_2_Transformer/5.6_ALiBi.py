# ALiBi 挺简单的，就是把偏置换成不可学习的，更加简单，能表示的范围也更加的远
# 之前考虑的都是推理和训练的长度一致，或者是transformer_xl针对于长序列拆成两个序列输入的问题改进
import math
import torch   
import torch.nn as nn

# ALiBi 位置偏置，就是相对位置对应的斜率
def get_alibi_slopes(num_heads):
    # ALiBi 的线性距离惩罚是核心机制，接下来就源论文的设计方法。主要记住偏置的线性惩罚就行了。
    # 输入就只需要几个头就行了，公式写的就是 斜率*相对位置。相对位置在类中用矩阵运算就行

    # 安全性检查一下
    if num_heads <= 0:
        raise ValueError("Number of heads must be a positive integer.")

    # 正式开始写，这里要注意两种情况，一种是 num_heads 是 2 的幂次方，一种是其他情况
    # 因为源论文的设计是线性函数的系数为2的幂次方

    # 首先判断 num_heads 是否是 2 的幂次方
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))

    if closest_power_of_2 == num_heads:
        # 第一种情况：Head 数量本身就是2的幂，可以直接生成等比数列
        # 例如4个 Head，start=2^(-8/4)=0.25,得到 [0.25, 0.0625, 0.015625, 0.00390625]
        # 这里的8是作者选定的斜率范围参数，并不代表必须使用8个 Head
        start = 2 ** (-8 / num_heads)

        # 生成 start^1、start^2、...、start^num_heads，直接用列表保存每个head的偏置
        slopes = [start ** (i + 1) for i in range(num_heads)]

    else:
        # 第二种情况：Head 数量不是2的幂，例如6个 Head
        # 数学上可以直接将n=6代入公式，但作者采用了另一套分配规则
        # 先生成4个 Head 的斜率，再从8-Head 序列中补充缺少的2个

        # 离6最近的2的幂是4，所以先生成4个head的斜率
        start = 2 ** (-8 / closest_power_of_2)
        slopes = [start ** (i + 1) for i in range(closest_power_of_2)]

        # 不够的head从下一个幂数开始算，例如6_head，最近的2次幂是8
        num_remaining_heads = num_heads - closest_power_of_2

        # 从8个head的斜率中补充缺少的head，只要 num_heads 不是 2 的幂，按照 ALiBi 作者的斜率分配规则，补充斜率时就取更大序列中的奇数次幂。
        start = 2 ** (-8 / (2 * closest_power_of_2))
        extra_slopes = [start ** (2 * i + 1) for i in range(num_remaining_heads)]

        slopes += extra_slopes

    return slopes


# ALiBi Multi-Head Self-Attention
class ALiBiMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()

        # d_model：输入 Token 的特征维度
        # num_heads：Attention Head 的数量
        # dropout：注意力权重的 Dropout 概率

        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("num_heads 必须为正数，且 d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 初始化 Q、K、V 的线性投影
        # ALiBi 不修改 Q、K、V 的计算方式，也不需要向输入 X 添加位置向量
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # 多头拼接后，还需要经过 W_O 将不同 Head 的信息重新融合
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # 根据 num_heads 生成固定斜率，每个 Head 对应一个斜率
        # 例如4个 Head 生成4个斜率，6个 Head 生成6个斜率
        slopes = torch.tensor(get_alibi_slopes(num_heads), dtype=torch.float32)

        # 斜率不需要通过训练学习，因此不使用 nn.Parameter
        # register_buffer 使 slopes 随模型移动到 CPU/GPU，但不会参与梯度更新
        # persistent=False 表示不保存到 state_dict，因为它可以根据 num_heads 重新生成
        self.register_buffer("slopes", slopes, persistent=False)

        # Dropout 用于训练时随机丢弃部分 Attention 权重
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, past_key_value=None, use_cache=False):
        # X：(B,L_q,d_model)，表示当前输入的 Token 特征
        # past_key_value：历史 K、V 缓存，第一次推理时为 None
        # use_cache：是否返回累积的 K、V，供下一次推理使用
        # 这里假设历史缓存与当前输入是同一序列中连续的 Token，暂不考虑 Padding Mask

        B, L_q, _ = X.shape

        # 根据输入 X 计算 Q、K、V
        # Self-Attention 中三者来自相同的输入，但使用不同的可学习投影矩阵
        # (B,L_q,d_model) -> (B,L_q,d_model)
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # 将 d_model 拆分成 num_heads 和 d_head
        # 例如 d_model=120，num_heads=6，则每个 Head 的特征维度 d_head=20
        # (B,L_q,d_model) -> (B,L_q,H,d_head)
        Q = Q.reshape(B, L_q, self.num_heads, self.d_head)
        K = K.reshape(B, L_q, self.num_heads, self.d_head)
        V = V.reshape(B, L_q, self.num_heads, self.d_head)

        # 将 Head 维度移动到 Token 维度前面，方便每个 Head 独立计算 Attention
        # (B,L_q,H,d_head) -> (B,H,L_q,d_head)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 前面已经对当前输入 X 计算了 Q、K、V，并完成多头拆分
        # 但如果正在使用 KV Cache，这里的 X 只包含本次新输入的 Token，因此 K、V 也只包含新 Token
        # 当前 Token 需要能够关注历史 Token，所以必须把历史缓存的 K、V 拼接回来
        if past_key_value is not None:
            # past_key_value 保存的是之前计算并缓存的 K、V，不是投影矩阵 W_K、W_V
            past_K, past_V = past_key_value

            # 历史 K：(B,H,L_past,d_head)，当前 K：(B,H,L_q,d_head)
            # dim=2 是 Token 维度，拼接后得到全部历史和当前 Token 的 K
            K = torch.cat([past_K, K], dim=2)  # (B,H,L_past+L_q,d_head)
            V = torch.cat([past_V, V], dim=2)  # (B,H,L_past+L_q,d_head)

        # Q 不需要拼接历史，因为当前只计算新 Token 的输出
        # K、V 则需要包含历史和当前 Token，才能让新 Query 关注之前的上下文

        # Key 包含历史和当前 Token，因此使用 KV Cache 时，L_k 可能大于 L_q
        L_k = K.shape[2]

        # 计算普通 Scaled Dot-Product Attention 的内容匹配分数
        # Q：(B,H,L_q,d_head)，K^T：(B,H,d_head,L_k)
        # 矩阵相乘得到 scores：(B,H,L_q,L_k)
        scores = Q @ K.transpose(-2, -1)

        # 除以 sqrt(d_head)，控制点积分数的尺度，避免 Softmax 过于尖锐
        scores = scores / math.sqrt(self.d_head)

        # 接下来才是 ALiBi 新增的位置偏置部分
        # 沿用 D[i,j]=Query位置-Key位置，正数表示过去，负数表示未来

        # Key 包含历史和当前 Token，所以其位置从0开始，一直排列到 L_k-1
        key_pos = torch.arange(L_k, device=X.device)

        # Query 只对应当前输入，所以起点应该是历史 Token 的数量，即 L_k-L_q
        # 例如历史有3个 Token，当前输入2个，则 key_pos=[0,1,2,3,4]，query_pos=[3,4]
        query_pos = torch.arange(L_k - L_q, L_k, device=X.device)

        # 构造相对距离矩阵
        # query_pos[:,None]：(L_q,1)，key_pos[None,:]：(1,L_k)
        # 广播相减得到 D：(L_q,L_k)，行对应 Query，列对应 Key
        D = query_pos[:, None] - key_pos[None, :]

        # 当前实现的是因果 Attention，只允许关注当前位置和过去的 Token
        # D>=0 对应当前及过去，D<0 对应未来，未来位置后面需要使用 Mask 屏蔽
        # 这里先将负数变成0，方便计算 ALiBi 的非负距离惩罚
        distance = D.clamp(min=0).to(dtype=scores.dtype)  # (L_q,L_k)

        # 每个 Head 有一个固定斜率，需要与整个相对距离矩阵相乘
        # slopes：(H,) -> (1,H,1,1)，让不同 Head 使用不同斜率
        # distance：(L_q,L_k)，后面会扩展成 (1,1,L_q,L_k)
        slopes = self.slopes.to(dtype=scores.dtype).view(1, self.num_heads, 1, 1)

        # 计算 ALiBi 的线性距离偏置：bias=-m_h*D
        # 斜率越大、距离越远，扣除的分数越多；不同 Head 的惩罚强度不同
        # slopes：(1,H,1,1)，distance：(1,1,L_q,L_k)
        # 广播相乘以后得到 bias：(1,H,L_q,L_k)
        bias = -slopes * distance[None, None, :, :]

        # 将 ALiBi 的位置偏置加到普通 Attention Score 上
        # scores：(B,H,L_q,L_k)，bias：(1,H,L_q,L_k)
        # 位置偏置在 Batch 之间共享，因此通过广播完成相加
        scores = scores + bias

        # 因果 Mask：禁止当前 Query 关注未来的 Key
        # 因为 D=Query位置-Key位置，所以 D<0 就是未来的位置
        # ALiBi 的距离惩罚不能代替 Mask，未来位置仍然需要设置为 -inf
        causal_mask = D < 0  # (L_q,L_k)

        # 将 Mask 扩展为 (1,1,L_q,L_k)，让所有 Batch 和 Head 使用相同的因果约束
        scores = scores.masked_fill(
            causal_mask[None, None, :, :],
            float("-inf")
        )

        # 沿 Key 维度进行 Softmax，将内容分数和位置偏置共同转换为注意力权重
        # ALiBi 在 Softmax 前减去线性距离惩罚，因此会影响最终各个 Key 的权重
        attn = torch.softmax(scores, dim=-1)  # (B,H,L_q,L_k)

        # 训练时对注意力权重使用 Dropout，推理时自动关闭
        attn_dropout = self.dropout(attn)

        # 使用注意力权重对 V 进行加权求和
        # (B,H,L_q,L_k) @ (B,H,L_k,d_head) -> (B,H,L_q,d_head)
        output = attn_dropout @ V

        # 将 Head 维度移动回 Token 维度后面
        # (B,H,L_q,d_head) -> (B,L_q,H,d_head)
        output = output.transpose(1, 2)

        # 将多个 Head 拼接起来，恢复 d_model 维表示
        # contiguous() 整理转置后的内存布局，再通过 reshape 合并 H 和 d_head
        output = output.contiguous().reshape(B, L_q, self.d_model)

        # 拼接后的多头结果再经过 W_O 投影
        output = self.W_O(output)  # (B,L_q,d_model)

        # 如果需要 KV Cache，就返回当前累积的 K、V，供下一次推理使用
        new_past_key_value = (K, V) if use_cache else None

        return output, attn, new_past_key_value


# 测试
if __name__ == "__main__":
    torch.manual_seed(42)

    # 先检查斜率生成函数的两种情况
    print("4个 Head 的斜率：", get_alibi_slopes(4))
    print("6个 Head 的斜率：", get_alibi_slopes(6))
    print("8个 Head 的斜率：", get_alibi_slopes(8))

    B = 2
    L = 6
    d_model = 120
    num_heads = 6

    # 使用6个 Head，顺便验证斜率生成函数中的第二种情况
    X = torch.randn(B, L, d_model)

    model = ALiBiMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.0
    )

    # 关闭 Dropout，方便比较完整序列和 KV Cache 的计算结果
    model.eval()

    # 1. 不使用 KV Cache，一次性处理完整序列
    with torch.no_grad():
        output, attn, _ = model(X)

    print("\n完整序列 Attention")
    print("输入形状：", X.shape)
    print("输出形状：", output.shape)  # (2,6,120)
    print("Attention 权重形状：", attn.shape)  # (2,6,6,6)
    print("当前模型的固定斜率：", model.slopes.tolist())

    # 因果 Attention 中，第一个 Query 只能关注位置0，其他未来位置的权重应该全部为0
    print("第一个 Query 的注意力权重：", attn[0, 0, 0])

    # 检查所有未来位置的注意力权重是否为0
    future_weights = torch.triu(attn, diagonal=1)
    assert torch.allclose(future_weights, torch.zeros_like(future_weights))

    # 2. 使用 KV Cache，将同一序列拆成两次输入
    with torch.no_grad():
        # 第一次输入前3个 Token，同时保存它们的 K、V
        output1, attn1, cache = model(X[:, :3], use_cache=True)

        # 第二次只输入后3个 Token，并读取之前缓存的 K、V
        output2, attn2, cache = model(
            X[:, 3:],
            past_key_value=cache,
            use_cache=True
        )

    print("\nKV Cache 推理")
    print("第一次输出形状：", output1.shape)  # (2,3,120)
    print("第二次输出形状：", output2.shape)  # (2,3,120)
    print("第二次 Attention 权重形状：", attn2.shape)  # (2,6,3,6)
    print("缓存 K 的形状：", cache[0].shape)  # (2,6,6,20)
    print("缓存 V 的形状：", cache[1].shape)  # (2,6,6,20)

    # 将两次输出拼接，并与一次性处理完整序列的输出比较
    # 因果 Mask 保证前面的 Query 不会关注未来，因此两种方式应该得到相同结果
    cached_output = torch.cat([output1, output2], dim=1)

    assert torch.allclose(cached_output, output, atol=1e-5, rtol=1e-5)

    print("\n因果 Mask 检查通过")
    print("KV Cache 与完整序列计算结果一致")
