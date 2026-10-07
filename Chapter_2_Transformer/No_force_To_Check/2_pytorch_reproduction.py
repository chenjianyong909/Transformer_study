# 用纯 Python 手写一个迷你版 PyTorch，模拟三件事：
#    1. 前向传播：张量做运算，同时搭建计算图
#    2. 反向传播：沿着计算图链式求导，算出每个参数的梯度
#    3. 梯度下降：用梯度更新参数数值

# 构建数据存储和张量存储的对象
'''
PyTorch 底层把「数据内存」和「张量视图」分开：
- StorageImpl：只管存原始数字（一块连续一维内存）
- TensorImpl：只管描述怎么看这块内存（形状、步幅、偏移）

分开的好处：切片、reshape 等视图操作不用拷贝数据，只改视图就行；
多个TensorImpl可以共享同一块StorageImpl，实现数据视图复用。

只有叶子节点的张量才会保存梯度，中间节点的梯度只在反向传播时中转，最终梯度只留在叶子张量上进行更新。
'''
# 数据节点
class StorageImpl:
    ''' 存储类，模拟CPU上一块连续内存 '''
    def __init__(self, data):
        self.data = data  # 一维列表，模拟连续内存，保存真实数值
        self.device = 'cpu'  # 标记数据存放设备

# 张量节点
class TensorImpl:
    '''
        张量视图类：通过 shape + stride + offset 控制"怎么从一维内存读出多维张量"
        👉 TensorImpl 就是我们代码里的【张量节点】
        所有变量 a,b,c、中间结果s1,s2、最终输出L 全部都是 TensorImpl实例

        三个核心视图参数：
        - shape：张量形状（几行几列）
        - stride：步幅（每维度走一步，需要在一维内存跳过几个元素）
        - offset：偏移量（从一维内存第几个元素开始读取）

        额外梯度相关成员：
        - requires_grad：是否需要追踪梯度、参与反向求导
        - grad_fn：指向【生成当前张量的运算Node算子】
          ✅ 若grad_fn=None：这是叶子张量（用户手动创建，非运算产物）
          ✅ 若grad_fn有值：这是中间张量（由某个加法/乘法算子前向计算得到，例如s1、s2）
        - grad：保存该张量的梯度值，同样是TensorImpl

        ============ 本次示例完整计算图（仅加法运算 L = (a+b)+(a+c)）============
        【张量节点】          【算子Node节点】        【张量节点】
             a ───┐
                  ├─▶ Add₁(加法算子) ──▶ s₁(中间张量) ───┐
             b ───┘                                      │
                                                         ├─▶ Add₃(加法算子) ──▶ L(最终输出张量)
             a ───┐                                      │
                  ├─▶ Add₂(加法算子) ──▶ s₂(中间张量) ───┘
             c ───┘

        对应每个算子登记的parents依赖：
        Add1.parents = [(ACCUMULATE_GRAD, a), (ACCUMULATE_GRAD, b)]
        Add2.parents = [(ACCUMULATE_GRAD, a), (ACCUMULATE_GRAD, c)]
        Add3.parents = [(Add1, s1), (Add2, s2)]

        张量与算子绑定关系：
        s1.grad_fn = Add1    说明s1由Add1运算生成
        s2.grad_fn = Add2    说明s2由Add2运算生成
        L.grad_fn  = Add3    说明L由Add3运算生成
        =====================================================================
        '''
    def __init__(self):
        self.shape = None          # 张量形状
        self.stride = None         # 张量步幅，PyTorch自动推导
        self.offset = 0            # 内存读取起始偏移
        self.storage_impl = None   # 指向原始内存，数据存放位置，也就是数据节点
        self.requires_grad = False # 是否开启梯度追踪
        self.grad_fn = None        # 生成本张量的运算节点(Node)，中间张量必有值，叶子张量为None
        self.grad = None           # 本张量对应的梯度张量，保存的类型是tensor

    def get_item(self, *indices):
        """
        根据多维索引取值，indices 支持任意维度
        公式：内存位置 = offset + Σ(indices[i] * stride[i])

        例子：
            二维张量 shape=[2,2],
            stride=[2,1] --> 行步幅2，列步幅1（换一行要向后跳2个元素，换一列跳1个）
            offset=0, storage_impl.data=[1,2,3,4]

          get_item(1, 0) → indices=(1,0)
            pos = 0 + 1*2 + 0*1 = 2
            返回 data[2] = 3
        """
        pos = self.offset          # 先加上内存起始偏移量

        # 遍历每一个维度索引，累加偏移
        for i in range(len(indices)):
            pos += indices[i] * self.stride[i]

        # 用计算出的一维内存下标取出对应数值
        return self.storage_impl.data[pos]

# 常量标记：代表上游输入是【叶子张量】
# 反向传播递归到此处终止，梯度直接写入 tensor.grad，不再继续往前追溯
ACCUMULATE_GRAD = "accumulate_grad"

# 算子节点
class Node:
    """
    【运算算子节点】：和TensorImpl（张量节点）区分开！
    每执行一次运算(加法/乘法等)，就生成一个Node算子，负责记录计算图依赖关系
    三个核心成员：
    1. backward_fn：反向求导函数，反向走到这个算子就调用它分发梯度
    2. parents：上游输入依赖列表，记录本次运算用到了哪些输入
       每一项格式是 (parent_type, parent_tensor)
       - parent_tensor：输入张量节点（可以是叶子a/b/c，也可以是中间张量s1/s2）
       - parent_type：如何反向处理这个输入：
         · parent_type=ACCUMULATE_GRAD → parent_tensor是叶子张量，梯度直接写入tensor.grad
         · parent_type=某个Node算子 → parent_tensor是中间张量，由该算子算出，递归调用算子反向继续往前传梯度
    3. cache：前向缓存，反向求导要复用的中间数据（矩阵乘法常用，加法暂时不用）

    ============ 对应上面计算图的parents拆解再说明 ============
             a ───┐
                  ├─▶ Add₁(加法算子) ──▶ s₁(中间张量) ───┐
             b ───┘                                      │
                                                         ├─▶ Add₃(加法算子) ──▶ L(最终输出张量)
             a ───┐                                      │
                  ├─▶ Add₂(加法算子) ──▶ s₂(中间张量) ───┘
             c ───┘
    Add3运算输入张量是s1、s2：
      parent_tensor第一个元素 = s1，parent_type = Add1（生成s1的算子）
      parent_tensor第二个元素 = s2，parent_type = Add2（生成s2的算子）
    所以 Add3.parents = [(Add1, s1), (Add2, s2)]
    反向遍历Add3.parents时：
      遇到Add1就调用Add1.backward_fn，把梯度继续分给a、b
      遇到Add2就调用Add2.backward_fn，把梯度继续分给a、c
    a被两条路径共用，梯度会累加，最终da=2
    ===========================================================
    """

    def __init__(self):
        # 加法自带求导系数恒为 1，不会缩放上游传来的梯度，直接透传 grad_output 对应的 storage_impl.data 即可
        # 自带求导逻辑为 grad_output × 另一侧输入值；乘法系数取自另一侧输入张量的 storage_impl.data，通过逐元素相乘缩放上游梯度，系数随输入动态变化
        # 使用 cache 可以通过键名绑定两个输入的角色，不受 parents 存储顺序干扰，例如运算`out=x*y`（x=3、y=4）时即便 parents 顺序颠倒，也能准确取出对应数值计算梯度；
        # 不使用 cache 只能依靠索引区分输入角色，若 parents 误存为`[y,x]`，张量本身数值没有变化，但梯度计算时会错误配对两侧乘数，最终结果出错。
        self.cache = {} 

        # parents存放运算输入依赖，格式：(上游处理标记/上游算子Node, 输入张量对象)
        self.parents = []
        self.backward_fn = None # 该计算节点反向传播需调用的函数，例如：add_backward
        # 反向传播最起点（损失标量自身）的初始梯度固定是 1，沿着计算图向左回溯，逐层分发梯度
        # 反向传播时中间张量仅中转梯度、不参与参数更新，梯度最终只留存于可训练的初始叶子张量，
        # 梯度下降只更新初始叶子节点参数，中间结果在下一轮前向计算时重新生成即可。

    # 该算子节点前向传播的时候用的
    @staticmethod
    def add_forward(x: TensorImpl, y: TensorImpl):
        """
        加法前向传播函数：
        功能：
         1. 数值运算：计算 x+y 得到输出数值
         2. 创建输出结果对应的中间张量节点（如s1、s2、L）
         3. 新建加法算子Node，登记parents依赖，构建计算图
         4. 将输出张量grad_fn绑定当前加法算子，反向时能顺着张量找到算子

        以 L=(a+b)+(a+c) 为例调用关系：
          s1 = Node.add_forward(a, b)   → s1.grad_fn = Add1
          s2 = Node.add_forward(a, c)   → s2.grad_fn = Add2
          L  = Node.add_forward(s1, s2)→ L.grad_fn  = Add3
        """
        # ========== 第1部分：数值前向计算 ==========
        # 取出两个输入张量底层一维内存数据
        x_data = x.storage_impl.data
        y_data = y.storage_impl.data
        # 逐元素相加得到加法结果一维数据
        result_data = [a + b for a, b in zip(x_data, y_data)]

        # ========== 第2部分：构建输出张量节点（TensorImpl，例如s1/s2/L） ==========
        result_storage = StorageImpl(result_data)
        result_tensor = TensorImpl()
        # 加法不改变张量形状、步幅、偏移，直接复用输入x配置
        result_tensor.shape = x.shape
        result_tensor.stride = x.stride
        result_tensor.offset = x.offset
        # 视图绑定自己的内存
        result_tensor.storage_impl = result_storage
        # 只要任意输入需要梯度，输出就需要梯度追踪
        result_tensor.requires_grad = x.requires_grad or y.requires_grad

        # ========== 第3部分：搭建计算图 ==========
        # 创建本次加法对应的算子节点
        add_node = Node()
        # 绑定加法专属反向函数
        add_node.backward_fn = add_backward

        def get_parent_entry(tensor: TensorImpl):
            """
            判断输入张量是叶子张量还是中间张量：
            规则：tensor.grad_fn 为空 → 用户手动创建叶子张量
                  tensor.grad_fn 不为空 → 该张量由之前某个算子Node运算产生（如s1由Add1算出）
            """
            if tensor.grad_fn is None:
                # 叶子张量：标记ACCUMULATE_GRAD
                return (ACCUMULATE_GRAD, tensor)
            else:
                # 中间张量：要继续调用生成它的算子反向传播梯度
                return (tensor.grad_fn, tensor)

        # 登记本次加法两个输入的依赖关系
        add_node.parents = [
            get_parent_entry(x),
            get_parent_entry(y)
        ]
        # 关键绑定：输出中间张量记住自己由哪个算子生成
        # 例如 s1.grad_fn = Add1；L.grad_fn = Add3
        result_tensor.grad_fn = add_node

        return result_tensor

    # 看完加法建议自己写乘法了
    @staticmethod
    def mul_forward(x: TensorImpl, y: TensorImpl):
        """
        乘法前向传播函数，和加法类似：计算-->构建计算图
        """

        # ====================== 前向计算 =======================
        x_data = x.storage_impl.data
        y_data = y.storage_impl.data
        result_data = [a * b for a, b in zip(x_data, y_data)]

        # ====================== 构建三种节点 =======================
        result_storage = StorageImpl(result_data) # 构建数据节点
        result_tensor = TensorImpl() # 构建张量节点
        # =========== 张量节点赋值 ==========
        result_tensor.shape = x.shape
        result_tensor.stride = x.stride
        result_tensor.offset = x.offset
        result_tensor.storage_impl = result_storage
        result_tensor.requires_grad = x.requires_grad or y.requires_grad
        result_tensor.grad = None # 乘法输出张量初始梯度为None

        mul_node = Node() # 构建乘法算子节点
        mul_node.backward_fn = mul_backward # 绑定乘法专属反向函数
        # ========== 新增关键缓存：保存两侧输入，供反向求导取用 ==========
        mul_node.cache = {"x": x, "y": y}

        # =============== 乘法算子初始化 =================
        def get_parent_entry(tensor: TensorImpl):
            if tensor.grad_fn is None:
                return (ACCUMULATE_GRAD, tensor)
            else:
                return (tensor.grad_fn, tensor)

        # 登记本次乘法两个输入的依赖关系
        mul_node.parents = [
            get_parent_entry(x),
            get_parent_entry(y)
        ]
        # 关键绑定：输出中间张量记住自己由哪个算子生成
        result_tensor.grad_fn = mul_node 

        return result_tensor


def add_backward(node: Node, grad_output: TensorImpl):
    """
    加法算子反向传播函数
    链式法则通用公式：
        ∂L / ∂输入 = (∂L / ∂当前输出) × (∂当前输出 / ∂输入)
    - grad_output 就是 ∂L/∂当前输出：上游算子传下来、损失对当前加法输出的梯度
    - 加法局部导数 ∂当前输出/∂输入 = 1（在只有加法操作的情况下，梯度永远为1）
    所以：每个输入接收梯度 = grad_output
    加法两侧梯度数值完全一致，直接复用传入的grad_output对象分发，不用新建梯度张量节点

    ============ 结合计算图走反向流程 ============
    初始启动反向：从L开始，人为传入初始梯度 grad_output 对应 ∂L/∂L = 1，调用Add3.backward_fn
    Add3遍历自己parents（也就是该算子节点的输入，即张量节点）：
      1. parent_type=Add1, parent_tensor=s1（s1是中间张量）
         → 调用Add1.backward_fn，把梯度1继续向前传给a、b
         → a第一次接收梯度：a.grad=None → a.grad=1
         → b第一次接收梯度：b.grad=None → b.grad=1
      2. parent_type=Add2, parent_tensor=s2（s2是中间张量）
         → 调用Add2.backward_fn，把梯度1继续向前传给a、c
         → a第二次接收梯度：a.grad已有旧值1，执行累加 a.grad=1+1=2
         → c第一次接收梯度：c.grad=None → c.grad=1
    ==============================================
    """
    # 遍历当前加法算子每一个输入依赖
    for parent_type, parent_tensor in node.parents:
        if parent_type == ACCUMULATE_GRAD:
            # 分支1：输入是叶子张量，梯度直接写入该张量grad
            if parent_tensor.grad is None:
                # 该叶子张量之前从未收到梯度，直接赋值
                parent_tensor.grad = grad_output
            else:
                # 该叶子张量已经有旧梯度，说明它被多条计算路径共用
                # 执行梯度累加：旧梯度 + 当前路径传来的新梯度
                parent_tensor.grad.storage_impl.data = [
                    old_g + new_g
                    for old_g, new_g in zip(parent_tensor.grad.storage_impl.data, grad_output.storage_impl.data)
                ]
        else:
            # 分支2：输入是中间张量（如s1/s2），parent_type就是生成该张量的上游算子Node
            # 梯度不能停在s1/s2，要继续递归调用上游算子backward_fn向前传播
            parent_node = parent_type
            parent_node.backward_fn(parent_node, grad_output)


def mul_backward(node: Node, grad_output: TensorImpl):
    """
    乘法反向传播函数：沿着计算图从右向左回溯，链式法则拆分梯度，分发给两个输入张量
    保留原有完整梯度流向示意图：
    ======== 示例：out = 2 * x * y（取值 x=3, y=4） ========
    【前向数据流｜严格左 → 右】
    const_2(值=2, requires_grad=False) ────┐
                                           ▼
    x(叶子=3, requires_grad=True) ─────▶ Mul1 ──▶ s₁(中间张量=6) ────┐
                                                                     ▼
    y(叶子=4, requires_grad=True) ───────────────────────────────▶ Mul2 ──▶ out(输出=24)

    绑定关系：
    s₁.grad_fn = Mul1
    out.grad_fn = Mul2

    【反向梯度流｜严格右 → 左，节点位置与前向完全对齐】
    const_2(丢弃梯度 ∂L/∂const₂=12) ◀────┐
                                         |
    x(叶子梯度 ∂L/∂x=8) ◀──────────── Mul1 ◀── s₁(接收梯度 ∂L/∂s₁ = y = 4) ◀───────────────┐
          Mul1内：入梯度=4, ∂L/∂x=4×2=8, ∂L/∂const₂=4×3=12                                  |
                                                                                         Mul2 ◀── out(初始梯度 ∂L/∂out=1)
    y(叶子梯度 ∂L/∂y=6) ◀──────────────────────────────────────────────────────────────────┘
          Mul2内：入梯度=1, ∂L/∂y=1×6=6, ∂L/∂s₁=1×4=4

    ======== 通用链式法则 ========
    前向：out = x * y
    上游传入：grad_output = ∂L/∂out
    ∂L/∂x = grad_output * y
    ∂L/∂y = grad_output * x

    关键设计解释：必须新建两组独立梯度张量节点
    1.乘法两侧梯度缩放系数分别是y、x，算出两组不同数值，不能共用同一个grad_output对象，复用会覆盖结果；
    2.叶子张量grad、上游算子入参只能接收TensorImpl格式，裸数值列表无法做梯度累加、链式回溯；
    3.新建临时梯度节点仅反向瞬时占用内存，分发完成自动回收，开销远小于篡改前向输出维度。
    优化说明：取消固定顺序grad_list索引配对，遍历父节点通过张量对象身份匹配cache内x/y实时算梯度，不受parents排列顺序影响，杜绝顺序错位风险

    执行步骤：
    1. 从算子缓存取出前向保存的两个输入 x、y
    2. 遍历parents，按张量归属实时计算对应梯度张量
    3. 分发梯度：
       - 叶子张量(ACCUMULATE_GRAD)：requires_grad=True 则累加梯度到 .grad
       - 中间张量：调用上游算子 backward_fn，继续向左传播梯度
    """
    # 从当前乘法算子缓存取出前向保存的两个输入张量，缓存依靠键名绑定角色，避免parents顺序颠倒出错
    # 当前算子节点的两个输入张量分别是 x、y，在forward过程中已经保存到 node.cache 里了
    x = node.cache["x"]
    y = node.cache["y"]

    # 取出上游梯度、两个输入原始一维数据，用于逐元素运算
    grad_out_data = grad_output.storage_impl.data
    x_data = x.storage_impl.data
    y_data = y.storage_impl.data

    # 沿计算图向左回溯，遍历父节点靠对象身份匹配归属、实时计算梯度下发，不再依赖固定索引列表
    for parent_type, parent_tensor in node.parents:
        # 必须得新建一个节点额外保存，因为两个parent_tensor的梯度数值不同，不能共用同一个grad_output对象，否则会覆盖掉另一侧的梯度
        # 加法就都一样了，无所谓的
        # 判断当前父张量是x还是y，套用对应链式求导公式
        if parent_tensor is x:
            #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            # 推导得到偏导表达式后，会带入运算当时`x、y`的实际数值计算最终梯度
            #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            # x的梯度 = 上游梯度逐元素 × y原始值
            cur_grad_data = [g * yi for g, yi in zip(grad_out_data, y_data)]
        else:
            # y的梯度 = 上游梯度逐元素 × x原始值
            cur_grad_data = [g * xi for g, xi in zip(grad_out_data, x_data)]
        
        # 封装成标准梯度张量实例
        cur_grad_storage = StorageImpl(cur_grad_data)
        current_grad = TensorImpl()
        current_grad.storage_impl = cur_grad_storage

        if parent_type == ACCUMULATE_GRAD:
            # 常量张量不需要保存梯度，跳过赋值流程
            if parent_tensor.requires_grad:
                if parent_tensor.grad is None:
                    # 首次接收梯度直接赋值
                    parent_tensor.grad = current_grad
                else:
                    # 多路复用张量执行梯度累加，禁止覆盖旧梯度
                    old_grad_data = parent_tensor.grad.storage_impl.data
                    new_grad_data = current_grad.storage_impl.data
                    parent_tensor.grad.storage_impl.data = [
                        old + new for old, new in zip(old_grad_data, new_grad_data)
                    ]
        else:
            # 中间张量继续向上游算子传递梯度
            upstream_op_node = parent_type
            upstream_op_node.backward_fn(upstream_op_node, current_grad)



if __name__ == "__main__":
    print("======== 测试 StorageImpl 和 TensorImpl ========")
    # 常数也要建立独立 TensorImpl 张量节点 + StorageImpl 数据节点
    # 创建一块连续内存存储
    storage = StorageImpl([1, 2, 3, 4])
    # 创建张量视图
    tensor = TensorImpl()
    tensor.shape = [2, 2]
    tensor.stride = [2, 1]
    tensor.offset = 0
    tensor.storage_impl = storage
    # 读取第1行第0列元素，预期输出3
    print(f"二维张量索引取值结果：{tensor.get_item(1, 0)}")

    print("\n======== 测试 Node 基础结构 ========")
    # 创建两个叶子张量x,y
    x = TensorImpl()
    y = TensorImpl()
    # 手动构造加法算子，输入x、y均为叶子张量
    add_node = Node()
    add_node.parents = [
        (ACCUMULATE_GRAD, x),
        (ACCUMULATE_GRAD, y)
    ]
    print("加法节点的父节点：", add_node.parents)

    print("\n======== 乘法完整前向+反向梯度传播测试 out = x * y ========")
    # 1.初始化叶子张量 x=3、y=4，开启梯度追踪（标量场景，底层一维列表长度为1）
    x = TensorImpl()
    x.storage_impl = StorageImpl([3])
    x.shape = [1]
    x.stride = [1]
    x.requires_grad = True

    y = TensorImpl()
    y.storage_impl = StorageImpl([4])
    y.shape = [1]
    y.stride = [1]
    y.requires_grad = True

    # 2.前向计算 out = x*y，自动构建计算图、算子缓存
    out = Node.mul_forward(x, y)
    print(f"前向计算结果 out=x*y：{out.storage_impl.data}") # 预期 [12]

    # 3.启动反向传播：损失L=out，初始梯度 ∂L/∂out=1
    init_grad_tensor = TensorImpl()
    init_grad_tensor.storage_impl = StorageImpl([1])
    out.grad_fn.backward_fn(out.grad_fn, init_grad_tensor)

    # 4.校验梯度理论值：∂L/∂x = y=4、∂L/∂y = x=3
    print(f"叶子张量x梯度（理论值4）：{x.grad.storage_impl.data}")
    print(f"叶子张量y梯度（理论值3）：{y.grad.storage_impl.data}")

    print("\n======== 验证parents顺序调换无配对错位（核心优化点测试）========")
    # 手动新建两组张量，颠倒parents传入顺序，验证对象身份匹配逻辑生效
    m1 = TensorImpl()
    m1.storage_impl = StorageImpl([2])
    m1.shape = [1]
    m1.stride = [1]
    m1.requires_grad = True

    m2 = TensorImpl()
    m2.storage_impl = StorageImpl([5])
    m2.shape = [1]
    m2.stride = [1]
    m2.requires_grad = True
    # 前向生成乘法算子后手动颠倒parents顺序
    out2 = Node.mul_forward(m1, m2)
    out2.grad_fn.parents = out2.grad_fn.parents[::-1]
    # 启动反向传播
    init_g2 = TensorImpl()
    init_g2.storage_impl = StorageImpl([1])
    out2.grad_fn.backward_fn(out2.grad_fn, init_g2)
    print(f"颠倒parents顺序后 m1梯度(理论5)：{m1.grad.storage_impl.data}")
    print(f"颠倒parents顺序后 m2梯度(理论2)：{m2.grad.storage_impl.data}")

'''
======== 测试 StorageImpl 和 TensorImpl ========
二维张量索引取值结果：3

======== 测试 Node 基础结构 ========
加法节点的父节点： [('accumulate_grad', <__main__.TensorImpl object at 0x7f5ee142b970>), ('accumulate_grad', <__main__.TensorImpl object at 0x7f5ee142b910>)]

======== 乘法完整前向+反向梯度传播测试 out = x * y ========
前向计算结果 out=x*y：[12]
叶子张量x梯度（理论值4）：[4]
叶子张量y梯度（理论值3）：[3]

======== 验证parents顺序调换无配对错位（核心优化点测试）========
颠倒parents顺序后 m1梯度(理论5)：[5]
颠倒parents顺序后 m2梯度(理论2)：[2]
'''