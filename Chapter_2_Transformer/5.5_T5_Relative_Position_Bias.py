
import torch
import torch.nn as nn
import math


# T5 相对位置分桶
def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    # relative_position：(L_q,L_k)，采用 Query位置-Key位置，正数表示过去，负数表示未来
    # num_buckets：可学习的位置偏置桶总数
    # max_distance：远距离分桶的参考上界，并不是实际距离的最大值

    if bidirectional:
        # 双向 Attention 需要区分过去和未来，所以将桶平分为两个方向
        n = num_buckets // 2

        # 负数（未来）使用前 n 个桶，正数（过去）使用后 n 个桶
        # 这里先记录方向偏移，后面只需要根据距离大小计算局部桶编号
        direction_offset = (relative_position > 0).to(torch.long) * n

        # 方向已经记录下来，因此取绝对值，统一处理距离大小
        relative_position = torch.abs(relative_position)

    else:
        # 因果 Attention 只看当前和过去，因此不需要划分方向
        n = num_buckets
        direction_offset = 0

        # 正数表示过去，负数表示未来；这里只保留非负距离，未来位置后面通过 Mask 屏蔽
        relative_position = relative_position.clamp(min=0)

    # 当前方向的一半桶用于精确距离，另一半桶用于远距离
    max_exact = n // 2

    # 判断哪些距离可以直接使用原始距离作为桶编号
    is_small = relative_position < max_exact

    # 远距离从 max_exact 开始计算，较小的距离暂时调整为 max_exact，避免后续出现 log(0)
    safe_distance = relative_position.float().clamp(min=max_exact)

    # 计算距离相对于远距离起点的倍数，例如 max_exact=16 时，距离32对应倍数2
    relative_ratio = safe_distance / max_exact

    # 使用对数压缩距离，让距离越远，相同数量的桶可以覆盖越大的距离范围
    log_distance = torch.log(relative_ratio)

    # 用整个远距离区间的对数跨度进行归一化，将 max_exact~max_distance 映射到0~1
    log_max = math.log(max_distance / max_exact)
    ratio = log_distance / log_max

    # 乘以剩余桶的数量，将归一化比例转换为远距离区间内部的编号
    far_index = ratio * (n - max_exact)

    # 桶编号必须是整数，因此向下取整并转换为 long 类型
    far_index = torch.floor(far_index).long()

    # 前 max_exact 个桶已经分配给近距离，所以远距离桶需要从 max_exact 开始编号
    far_bucket = max_exact + far_index

    # 超过 max_distance 的距离统一使用最后一个桶，避免 Embedding 索引越界
    far_bucket = far_bucket.clamp(max=n - 1)

    # 近距离直接使用实际距离，远距离使用对数桶编号，最后加上方向偏移
    return direction_offset + torch.where(is_small, relative_position, far_bucket)


# T5 相对位置多头 Self-Attention
class T5RelativeAttention(nn.Module):

    def __init__(self, d_model, num_heads, num_buckets=32, max_distance=128, bidirectional=True):
        super().__init__()

        # d_model 必须能够平均分给所有 Head，因为后面需要将 d_model 拆成 num_heads 个 d_head
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("num_heads 必须为正数，且 d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        # Self-Attention 中，Q、K、V 都由输入 X 经过不同的可学习矩阵得到
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # 多头 Attention 的输出需要重新投影回 d_model
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # T5 不再为每个距离学习 d_model 维向量，而是为每个桶、每个 Head 学习一个标量偏置
        # 例如 nn.Embedding(32,4)，表示32个桶，每个桶存储4个 Head 对应的位置偏置
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def forward(self, X):
        # X：(B,L,d_model)，这里实现普通 Self-Attention，暂时不考虑历史 KV Cache
        B, L, d_model = X.shape

        # 1. 根据输入 X 计算 Q、K、V，形状与输入保持一致
        Q = self.W_Q(X)  # (B,L,d_model)
        K = self.W_K(X)  # (B,L,d_model)
        V = self.W_V(X)  # (B,L,d_model)

        # 2. 将 d_model 拆分成 num_heads 和 d_head，让每个 Head 拥有独立的特征
        Q = Q.reshape(B, L, self.num_heads, self.d_head)
        K = K.reshape(B, L, self.num_heads, self.d_head)
        V = V.reshape(B, L, self.num_heads, self.d_head)

        # 3. 将 Head 移到 Token 维度前面，方便每个 Head 独立计算 QK^T
        # (B,L,H,d_head) -> (B,H,L,d_head)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 4. 计算普通 Multi-Head Attention 的内容匹配分数
        # (B,H,L,d_head) @ (B,H,d_head,L) -> (B,H,L,L)
        scores = Q @ K.transpose(-2, -1)

        # 这里采用之前学过的标准缩放方式，避免点积分数随 d_head 增大而过大
        scores = scores / math.sqrt(self.d_head)

        # 5. 接下来才是 T5 新增的位置偏置部分
        # 每个 Query-Key 对需要根据相对距离确定一个桶，因此先计算相对距离矩阵
        position = torch.arange(L, device=X.device)

        # 行对应 Query，列对应 Key，D[i,j]=i-j，正数表示过去，负数表示未来
        D = position[:, None] - position[None, :]  # (L,L)

        # 6. 将每个相对距离映射为桶编号
        # 分桶函数只负责产生整数索引，输出形状与相对距离矩阵相同
        bucket = relative_position_bucket(
            D,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )  # (L,L)

        # 7. 根据桶编号查表，得到每个 Query-Key 对在不同 Head 中的可学习位置偏置
        # 每个桶对应 H 个标量，因此 (L,L) -> (L,L,H)
        bias = self.relative_attention_bias(bucket)  # (L,L,H)

        # 8. 现在 scores 是 (B,H,L,L)，bias 是 (L,L,H)，Head 维度的位置不一致
        # 因此将 bias 的第2维 H 移到最前面，使 Head、Query、Key 的顺序与 scores 一致
        bias = bias.permute(2, 0, 1)  # (H,L,L)

        # 9. 不同 Batch 可以共用同一套位置偏置，所以增加大小为1的 Batch 维度进行广播
        bias = bias.unsqueeze(0)  # (1,H,L,L)

        # 10. 将位置偏置直接加到对应 Head、Query、Key 的内容分数上
        # scores：(B,H,L,L)，bias：(1,H,L,L)，相加后仍然为 (B,H,L,L)
        scores = scores + bias

        # 11. 如果是因果 Attention，需要额外屏蔽未来位置，位置分桶本身不能代替 Mask
        if not self.bidirectional:
            # 当前 D=Query位置-Key位置，所以 D<0 表示 Key 位于 Query 的未来
            causal_mask = D >= 0  # (L,L)

            # 增加 Batch 和 Head 维度，让所有 Batch、Head 共用同一个因果 Mask
            scores = scores.masked_fill(
                ~causal_mask[None, None, :, :],
                float("-inf")
            )

        # 12. 沿 Key 维度进行 Softmax，让每个 Query 对所有可关注 Key 的权重之和为1
        attn = torch.softmax(scores, dim=-1)  # (B,H,L,L)

        # 13. 使用注意力权重对 V 加权求和，每个 Head 得到自己的输出
        output = attn @ V  # (B,H,L,d_head)

        # 14. 多头输出需要拼接，因此先将 Head 维度交换回 Token 维度后面
        output = output.transpose(1, 2)  # (B,L,H,d_head)

        # 15. 将 H 和 d_head 合并，恢复为 d_model 维表示
        # contiguous() 用于整理转置后的内存布局，方便后续 reshape
        output = output.contiguous().reshape(B, L, self.d_model)  # (B,L,d_model)

        # 16. 拼接后的多头表示再经过 W_O 投影，得到最终 Attention 输出
        output = self.W_O(output)  # (B,L,d_model)

        return output, attn


# 测试
if __name__ == "__main__":

    B = 2
    L = 5
    d_model = 128
    num_heads = 4

    X = torch.randn(B, L, d_model)

    # 双向 Self-Attention：Query 可以关注过去和未来
    model = T5RelativeAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_buckets=32,
        max_distance=128,
        bidirectional=True
    )

    output, attn = model(X)

    print("双向 Attention 输出：", output.shape)  # (2,5,128)
    print("双向 Attention 权重：", attn.shape)    # (2,4,5,5)

    # 因果 Self-Attention：Query 只能关注自身和过去
    causal_model = T5RelativeAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_buckets=32,
        max_distance=128,
        bidirectional=False
    )

    output, attn = causal_model(X)

    print("因果 Attention 输出：", output.shape)  # (2,5,128)
    print("因果 Attention 权重：", attn.shape)    # (2,4,5,5)

    # 第一行 Query 对未来 Key 的注意力权重应该全部为0
    print("第一个 Query 的注意力权重：", attn[0, 0, 0])
