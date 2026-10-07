import torch
import torch.nn as nn

# 将具有三个特征的样本作为输入，输出为两个特征
x = torch.tensor([10.0,20.0,30.0]) # x.shape = [3]，当前是一个一维Tensor，没有显式的batch维度
# [[10.0,20.0,30.0]]，这样写shape才是[1,3]，即[batch_size, input_features]

# 既然输出特征为2，根据矩阵运算来说相当[1,3] * [3,2] = [1,2]
# 因此计算时需要一个shape=[3,2]的矩阵；但这里按照PyTorch nn.Linear的方式存储W：
#   W.shape = [out_features, in_features] = [2,3]
#   W的每一行对应一个输出特征所使用的一组权重。
#   输入有3个特征，每个输出都需要读取这3个特征，所以每个输出需要3个权重；
#   一共有2个输出，因此W中有2组权重，即W.shape=[2,3]。
#   真正计算时使用W.T，此时W.T.shape=[3,2]。
# 另外，线性层还会有一个偏置b，b.shape = [2]，每个输出特征对应一个偏置。

W = torch.tensor([
    [1.0,2.0,3.0],
    [7.0,8.0,9.0],
])

b = torch.tensor([1.0,2.0]) # [output_features] = [2]

y = x @ W.t() + b # y = x @ W + b 会报错，因为[3] @ [2,3]不符合矩阵乘法规则
print(y) # tensor([141., 502.])，输出shape为[2]，即两个特征；如果x.shape=[1,3]，输出才是[1,2]
# PyTorch nn.Linear约定weight按照[out_features, in_features]存储，计算时使用weight.T。

# tensor的广播机制常见于逐元素运算中；
# 对于矩阵乘法@，真正参与矩阵乘法的维度必须满足矩阵乘法规则，不能靠广播解决；
# 但对于高维Tensor，前面的batch维度仍然可以发生广播，但是最后两维度一定得符合矩阵运算法则。
# 比如：[...,2,3] @ [...,3,4] = [...,2,4]，前面的batch维度可以广播，但最后两维必须满足矩阵乘法规则。
# 广播原理：
# 从最右边开始对齐维度
# 每一维满足下面任意一个条件，就可以广播：
#           两边相等
#           其中一个维度是 1
#           某个 Tensor 缺这一维，等价于左边补 1
#           只要有一维不满足条件，整个广播失败，停止剩余维度的广播
#           广播时不是补 0，而是把原来的值沿对应维度重复使用
# 例如：[3]和[2,3]运算时，[3]可以理解为[1,3]，再沿第0维复制成[2,3]
print('x.shape',x.shape) # x.shape torch.Size([3])，因为x本身就是一维Tensor，不存在batch维度
print('W.shape',W.shape) # W.shape torch.Size([2, 3])
print('b.shape',b.shape) # b.shape torch.Size([2])
print('y2 = ',x * W ) # y2 =  tensor([[ 10.,  40.,  90.],[ 70., 160., 270.]])
# y2 = x * W + b 会报错， x * W的结果是[2,3]，b是[2]，不满足广播条件
print('y3 = ',((x * W).T + b).T) # 这样的话可以利用广播进行计算了，但这只是广播示例，不等价于Linear的计算

# 了解了tensor以后，现在就开始使用nn.Linear来实现线性层的功能
# ================================== Linear层 ===================================
# 问题引入：假设现在我的输入有3个特征，输出有2个特征，那么我该如何使用nn.Linear来实现呢？
# nn.Linear的参数：
#   in_features: 输入特征的维度
#   out_features: 输出特征的维度
#   bias: 是否使用偏置，默认是True
# nn.Linear的本质： y = x @ weight.T + bias
linear_layer = nn.Linear(in_features=3, out_features=2, bias=True) # 定义一个线性层的输入和输出维度
# 线性层的权重矩阵W和偏置b在创建时都会被自动初始化
# 线性层的权重矩阵W和偏置b是可以通过linear_layer.weight和linear_layer.bias来访问的
print('linear_layer.weight',linear_layer.weight)
x_batch = torch.tensor([
    [10.0, 20.0, 30.0],   # 第1个样本
    [40.0, 50.0, 60.0],   # 第2个样本
])
print('x_batch.shape',x_batch.shape) # x_batch.shape torch.Size([2, 3])
print('linear_layer(x_batch)',linear_layer(x_batch)) # 利用定义好的线性层进行计算


# ================================ Linear层的参数初始化 ==========================================

# nn.Linear创建以后，weight和bias必须先有初始值才能开始训练。weight通常使用随机初始化，
# 目的是打破不同输出神经元之间的对称性。例如两个神经元如果weight完全相同：
#   y1 = w11*x1 + w12*x2 + w13*x3 + b1
#   y2 = w21*x1 + w22*x2 + w23*x3 + b2
# 那么它们会进行相同的计算并得到相同方向的参数更新，很难学习不同的特征。

# nn.Linear创建时会自动初始化weight和bias。默认weight最终等价于：
#   W ~ Uniform(-bound, bound)，bound = 1 / sqrt(fan_in)
# bias=True时，bias也使用相同bound范围的均匀分布初始化。


# ----------------------------- 初始化主要控制两个统计量 -----------------------------

# 初始化主要控制：
#   E[W] ≈ 0：让weight整体不偏正也不偏负，避免初始化人为引入方向偏置；
#   Var(W)：控制weight的数值尺度，从而控制Linear对输入信号的放大/缩小程度。
# 期望描述“中心位置”，方差描述“围绕中心的波动尺度”。

# 对单个Linear输出：
#   Z = w1*X1 + w2*X2 + ... + wn*Xn，n = fan_in
#   Var(Z) = E[Z²] - E[Z]² ≈ E[Z²](不带激活函数的分析，带激活函数的会放在下面分析)
#
#   E[Z²] = E[(w1*X1 + w2*X2 + ... + wn*Xn)²] = ΣE[wi²*Xi²] + 2ΣE[wi*Xi*wj*Xj]
#   ≈ ΣE[wi²]E[Xi²] = Var(W) * ΣE[Xi²]， E[wi²] = Var[wi] + E[wi]² -> E[wi] = 0 -> E[wi²] = Var[W]
#
#   令：qx = (1 / fan_in) * ΣE[Xi²]，用来表示输输入的平均偏移（认为定义，具体不细究了，先搞明白下一步就行）
#   Var(Z) ≈ E[Z²] ≈ ΣE[wi²]E[Xi²] = Var(W) * ΣE[Xi²]  = fan_in * Var(W) * qx
#   
#   这边就直接近似为 Var(output) = fan_in * Var[W] * Var(input)

# ----------------------------- 为什么希望Var(output) / Var(input)比例接近1？ -----------------------------

# 初始化并不是要求方差本身必须等于1，而是希望：Var(output) / Var(input) ≈ 1
# 即随机初始化本身不要系统性地放大或缩小信号。
#
# 原因是上一层输出会继续成为下一层输入，如果每层初始增益都>1或<1，这种尺度变化会随着网络深度连续累积。
# 初始化的作用只是给模型提供一个数值稳定的起点；该条件只用于初始化阶段。
#
# 训练开始以后：
#   W1 = W0 - lr * grad(W0)
# 参数不会继续被强制满足初始化时的方差关系，真正需要的特征增强或抑制由loss和梯度下降自己学习。


# ======================================================================================
# （1）不接激活函数：只看Linear
# ======================================================================================

# 假设网络为：
#   X0 -> Linear1 -> X1 -> Linear2 -> X2 -> Linear3 -> X3
#
# 初始化需要同时考虑：
#   ① 前向传播的特征尺度
#   ② 反向传播的梯度尺度


# ---------------- 前向传播 ----------------

# PyTorch写法： Y = X @ W.T
#
# 前面已经得到：Var(Y) / Var(X) ≈ fan_in * Var(W)。为了让前向信号的初始尺度尽量稳定：fan_in * Var(W) ≈ 1
#
# 因此：
#   Var(W) ≈ 1 / fan_in
#   Std(W) ≈ 1 / sqrt(fan_in)
#
# 工程含义：fan_in越大，一个输出需要汇集的输入越多，因此单个weight要相应缩小，
# 避免网络仅仅因为输入维度变宽，就让中间特征的数值尺度随之增大。


# ---------------- 反向传播 ----------------

# 前向：Y = X @ W.T
#
# PyTorch行向量形式的反向：grad_X = grad_Y @ W
#
# 例如两个输出：
#   y1 = w11*x1 + w12*x2 + w13*x3
#   y2 = w21*x1 + w22*x2 + w23*x3
#
# x1同时影响y1和y2，所以：
#   grad_x1 = grad_y1*w11 + grad_y2*w21
#
# 如果共有fan_out个输出：
#   grad_x1 = Σ(grad_yj * wj1)，j = 1...fan_out
#
# 因此和前向传播类似：
#   Var(grad_X) / Var(grad_Y) ≈ fan_out * Var(W)
#
# 为了让反向梯度的初始尺度尽量稳定：
#   fan_out * Var(W) ≈ 1
#
# 得到：
#   Var(W) ≈ 1 / fan_outv
#
# 所以纯Linear情况下：
#   前向传播希望 Var(W) ≈ 1 / fan_in
#   反向传播希望 Var(W) ≈ 1 / fan_out
#
# fan_in = fan_out时两者一致；fan_in != fan_out时无法同时严格满足。


# ============================= Xavier / Glorot 初始化 ==============================

# Xavier就是在前向和反向两个要求之间取折中：
#   fan_avg = (fan_in + fan_out) / 2
#
#   Var(W) ≈ 1 / fan_avg = 2 / (fan_in + fan_out)
#   Std(W) = sqrt(2 / (fan_in + fan_out))


# ---------------- Xavier Normal ----------------

#   正态分布：W ~ Normal(
#               mean = 0,
#               std = sqrt(2 / (fan_in + fan_out))
#            )

weight = torch.empty(2, 3)
nn.init.xavier_normal_(weight)


# ---------------- Xavier Uniform ----------------

#   均匀分布：Uniform(a, b)的方差：Var(W) = (b-a)² / 12
#       由于要求E[X] = 0 ，所以均匀分布就是Uniform(-b,b)
# Xavier要求：
#   b² / 3 = 2 / (fan_in + fan_out)
#
# 所以：
#   b = sqrt(6 / (fan_in + fan_out))
#
# 即：
#   W ~ Uniform(
#       -sqrt(6/(fan_in+fan_out)),
#        sqrt(6/(fan_in+fan_out))
#   )

weight = torch.empty(2, 3)
nn.init.xavier_uniform_(weight)


# ======================================================================================
# （2）接激活函数：Linear + Activation
# ======================================================================================

# 上面分析的是Linear本身，但实际网络通常是：
#   X -> Linear -> ReLU -> Linear -> ReLU -> ...
#
# 此时除了Linear，激活函数也会改变信号尺度，因此初始化必须考虑整个Linear + Activation。

# ---------------- ReLU为什么需要重新分析？ ----------------

# ReLU：
#   Y = ReLU(Z) = max(0, Z)
#
# 例如：
#   Z = [-3,-2,-1,1,2,3] -> E[Z] = (-3 + -2 + -1 + 1 + 2 + 3)/6 = 0 (对称的分布找中心对称的点，非对称用定义)
#   Y = [ 0, 0, 0,1,2,3] -> E[Y] = (1 + 2 + 3)/6 = 1
#
# ReLU把所有负数变成0，因此Y不再关于0对称，并且E[Y] > 0。
# 此时：
#   Var(Y) = E[Y²] - E[Y]²
#
# 所以不能再像纯Linear那样使用：
#   Var(Y) ≈ E[Y²]


# ---------------- 为什么改用二阶矩E[Y²]？ ----------------

# 因为ReLU输出Y会直接成为下一层Linear的输入，而下一层Linear的输出平方尺度直接由E[Y²]决定：
#   Z_next = Σ(wi * Yi)
#   E[Z_next²] ≈ fan_in * E[W²] * E[Y²]
#
# 所以E[Y²]不是为了方便随便换的指标，而是它本身就直接决定下一层Linear的信号平方尺度。


# ---------------- ReLU对二阶矩的影响 ----------------

# 如果Z近似关于0对称，那么+z和-z出现的统计概率相同，并且：
#   (+z)² = (-z)²
#
# 所以正半边和负半边对E[Z²]的贡献大致各占一半。
# ReLU把负半边全部清零，因此：
#
#   E[Y²] = E[ReLU(Z)²] ≈ 1/2 * E[Z²]
#
# 即ReLU会使二阶信号尺度大约减少一半。


# ---------------- Kaiming / He 初始化 ----------------

# 如果仍使用纯Linear的：Var(W) ≈ 1 / fan_in。目的是保证经过线性变换后的Z，二阶矩和输入X保持一致。
#
# Linear虽然能够保持Z的信号尺度，但经过ReLU以后只剩约一半。因此权重方差必须提前放大，补偿ReLU带来的信号衰减。
#
# 线性变换的二阶矩关系：X -> Linear -> Z

#   E[Z²] = fan_in * Var(W) * E[X²]

# 再套上ReLU(X-> Linear -> Z -> ReLU -> Y)：Y = ReLU(Z) = max(0, Z)
#   E[Y²] = E[ReLU(Z)²] ≈ 1/2 * fan_in * Var(W) * E[X²]
#
# 我们的目标：经过 Linear + ReLU 整套操作后，输出二阶矩 ≈ 输入二阶矩

# E[Y²]  ≈ 1/2 * fan_in * Var(W) * E[X²] -> E[Y²] ≈ E[X²]
# 故：  1/2 * fan_in * Var(W) ≈ 1

# 求解权重方差：
#   Var(W) ≈ 2 / fan_in
# 标准差：
#   Std(W) ≈ sqrt(2 / fan_in)
#
# 这就是Kaiming / He针对ReLU时sqrt(2/fan_in)的来源：
#   纯Linear只需要补偿fan_in个输入的累加；
#   Linear + ReLU还要额外补偿ReLU清除负半边造成的二阶信号损失。

# ========== 前向传播（信号尺度） ==========
# 前向：X -> Linear -> Z -> ReLU -> Y
# Z = X @ W.T -> Y = ReLU(Z)
# Var(W) ≈ 2 / fan_in

# ========== 反向传播（梯度尺度） ==========
# 前向公式 Y = ReLU(X @ W.T)
# 反向求输入梯度：grad_X = grad_Y @ W * I(Z>0)
# I(Z>0) 是ReLU的梯度掩码：Z>0时取1；Z≤0梯度直接置0

# 举简单例子，设fan_out个输出神经元：
# z1 = w11*x1 + w12*x2 + ...
# z2 = w21*x1 + w22*x2 + ...
# y1 = ReLU(z1), y2 = ReLU(z2)
#
# x1同时影响所有fan_out个输出y_j，求x1梯度：
# grad_x1 = Σ( grad_yj * wj1 * I(z_j>0) ), j = 1...fan_out
#
# 和前向对称，I(Z>0)这个掩码，在Z对称假设下，大约一半位置为0。
# 所以梯度经过ReLU之后，梯度的二阶矩同样会折半：
# Var(grad_X) / Var(grad_Y) ≈ 1/2 * fan_out * Var(W)
#
# 目标：稳定反向梯度尺度，防止梯度爆炸/消失
# 令 Var(grad_X) / Var(grad_Y) ≈ 1
#
# 得到约束：
#   0.5 * fan_out * Var(W) ≈ 1
# 求解：
#   Var(W) ≈ 2 / fan_out
#   Std(W) ≈ sqrt(2 / fan_out)

# fan_in != fan_out时，前向和反向无法同时满足约束条件。Kaiming/He初始化二选一，用户指定优先稳定前向 或是 优先稳定反向。

# ---------------- Kaiming Normal ----------------
# W ~ Normal(mean=0, std^2 = 2/fan_in)
#
# 即：
#   W ~ Normal(0, sqrt(2 / fan_in))
weight = torch.empty(2, 3)
nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="relu")

# ---------------- Kaiming Uniform ----------------
# 均匀分布：Uniform(a, b)的方差：Var(W) = (b-a)² / 12
#   由于要求E[X] = 0，均匀分布为Uniform(-b,b)
# Kaiming要求：
#   b² / 3 = 2 / fan_in
#
# 所以：
#   b = sqrt(6 / fan_in)
#
# 即：
#   W ~ Uniform(
#       -sqrt(6 / fan_in),
#        sqrt(6 / fan_in)
#   )
weight = torch.empty(2, 3)
nn.init.kaiming_uniform_(weight, mode="fan_in", nonlinearity="relu")
