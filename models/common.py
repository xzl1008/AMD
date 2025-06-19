import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / (self.affine_weight + self.eps * self.eps)[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x


class MDM(nn.Module):
    def __init__(self, input_shape, k=3, c=2, layernorm=True):
        super(MDM, self).__init__()
        self.seq_len = input_shape[0]
        self.k = k
        if self.k > 0:
            self.k_list = [c ** i for i in range(k, 0, -1)]
            self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])   # 公式(3)
            """
                nn.ModuleList 是 PyTorch 提供的一个容器，用来存放一系列子模块（nn.Module）。
                与普通 Python list 的区别在于，ModuleList 中的子模块会自动被注册到父模块中，其参数会在调用 .parameters()、.cuda() 等方法时被正确处理。
                使用 ModuleList 可以方便地在循环或列表推导式中生成多个子模块，保证它们都被视为模型的一部分；
                若仅用普通 list 存放这些子模块，则它们的参数不会被自动注册，也无法参与训练。
            """
            """
                AvgPool1d的参数作用：
                    kernel_size：指定池化窗口的长度，也就是每次取平均值时覆盖的时间步数或元素个数。如果 k=4，表示每次对长度为 4 的片段求平均。
                    stride：指定窗口滑动的步长，即池化窗口在序列上移动的距离。当 stride=k 时，每次窗口移动 k 个元素，相邻窗口没有重叠（等价于“分块”求平均）；如果 stride 小于 kernel_size，窗口会重叠，输出序列更长。            
                    因此，nn.AvgPool1d(kernel_size=k, stride=k) 会以长度为 k 的窗对输入序列做平均，每次移动 k 个位置，输出长度缩短为输入长度的 1/k，适合对序列做非重叠的降采样。
            """
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.seq_len // k, self.seq_len // k),
                                  nn.GELU(),
                                  nn.Linear(self.seq_len // k, self.seq_len * c // k),
                                  )
                    for k in self.k_list
                ]
            )
            """
            主要作用是为不同下采样比例准备对应的线性变换层：
                这里 self.k_list 保存了若干不同的池化系数（例如 [8, 4, 2]），与之对应的 self.avg_pools 是多个平均池化层。
                self.linears 中的每个元素都是一个 nn.Sequential 模块，包含两个 nn.Linear 和一个 nn.GELU 激活层：
                    1. 第一个 nn.Linear：输入和输出维度均为 self.seq_len // k，相当于在池化后的序列长度上做一次线性变换。
                       下采样后，序列长度从 seq_len 变为 seq_len // k。nn.Linear(self.seq_len // k, self.seq_len // k) 
                       就是对这段较短序列做一次全连接变换，相当于用一个形状为 (seq_len//k, seq_len//k) 的矩阵重新混合这些时间步的值。
                    2. nn.GELU：非线性激活。
                        模型使用 nn.GELU() 作为非线性激活。GELU（Gaussian Error Linear Unit）根据输入值的大小平滑地决定通过多少信号：
                            当 x>0 时，输出接近于x但略小）；
                            当 x<0 时，输出会被抑制得更厉害，但不像 ReLU 那样直接截断为 0。
                        GELU 的平滑性在一些任务中能带来更好的梯度流动效果，因此常用于 Transformer 等模型中。
                    3. 第二个 nn.Linear：将长度 self.seq_len // k 的表示映射为更长的 self.seq_len * c // k，用于与下一层（池化系数较小的层）对齐。
                
                这样在 forward 中，就可以先对输入做多尺度平均池化，再利用这些线性层把较短序列的特征映射回更长的序列，与原序列进行逐层相加，形成自上而下的多尺度融合。
            """
        self.layernorm = layernorm
        if self.layernorm:
            # 参数用来指定在展平后的维度数，使批归一化作用于整段序列与所有特征的组合，能够对输入的整体分布进行标准化。
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])

    def forward(self, x):
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        if self.k == 0:   # 当 k=0 时，说明没有需要执行的池化或线性混合操作，MDM 就直接返回输入 x。因此 k=0 表示关闭多尺度混合模块，使该层退化为（可选的）BatchNorm 之后的恒等映射。
            return x
        # x [batch_size, feature_num, seq_len]
        sample_x = []
        for i, k in enumerate(self.k_list):
            sample_x.append(self.avg_pools[i](x))
        sample_x.append(x)
        n = len(sample_x)
        for i in range(n - 1):   # 公式(4)   bottom_up融合（相加）
            tmp = self.linears[i](sample_x[i])
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)
        # [batch_size, feature_num, seq_len]
        # self.linears中使用k的顺序是[8,4,2]，c=2，所以最后一个nn.Linear输出的是self.seq_len * 2 // 2，也就是和输入相同，完成融合
        return sample_x[n - 1]


class DDI(nn.Module):
    """
    DDI 块的作用可以理解为：
        1. 在时间维度上按照 patch 大小滑动窗口，利用最近一个窗口的信息去“补全”或“更新”当前窗口；
        2. 如果 alpha 不为 0，则对每个窗口的特征维度再进行一次全连接映射，实现不同特征之间的交互；
        3. 返回的张量与输入形状一致，但内容已经经过时序和跨特征的混合处理。
    """
    def __init__(self, input_shape, dropout=0.2, patch=12, alpha=0.0, layernorm=True,
                 top_k=3, n_hop=1, heads=4):
        super(DDI, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.input_shape = input_shape
        if alpha > 0.0:    # alpha 大于 0 时构建图神经网络相关模块
            self.ff_dim = 2 ** math.ceil(math.log2(self.input_shape[-1]))
            if self.ff_dim % heads != 0:
                self.ff_dim = heads * math.ceil(self.ff_dim / heads)
            self.heads = heads
            self.channel_embed = nn.Linear(1, self.ff_dim)
            self.q_proj = nn.Linear(self.ff_dim, self.ff_dim)
            self.k_proj = nn.Linear(self.ff_dim, self.ff_dim)
            self.v_proj = nn.Linear(self.ff_dim, self.ff_dim)
            self.msg_mlp = nn.Sequential(
                nn.Linear(self.ff_dim, self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, 1)
            )
            self.top_k = top_k
            self.n_hop = n_hop

        self.n_history = 1    # 固定为 1，表示每次聚合只参考前一个窗口
        self.alpha = alpha
        self.patch = patch

        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        if self.alpha > 0.0:
            self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])

        self.agg = nn.Linear(self.n_history * self.patch, self.patch)    # 线性层，用于将长度为 n_history * patch 的历史序列压缩到 patch，即进行时间维度的聚合。
        self.dropout_t = nn.Dropout(dropout)

    def forward(self, x):
        # [batch_size, feature_num, seq_len]
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        output = torch.zeros_like(x)
        output[:, :, :self.n_history * self.patch] = x[:, :, :self.n_history * self.patch].clone()    # 将最前面的历史部分直接复制给 output。
        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            # input [batch_size, feature_num, self.n_history * patch]
            input = output[:, :, i - self.n_history * self.patch: i]
            # input [batch_size, feature_num, self.n_history * patch]
            input = self.norm1(torch.flatten(input, 1, -1)).reshape(input.shape)
            # aggregation
            # [batch_size, feature_num, patch]
            input = F.gelu(self.agg(input))  # self.n_history * patch -> patch
            input = self.dropout_t(input)
            # input [batch_size, feature_num, patch]
            # input = torch.squeeze(input, dim=-1)
            tmp = input + x[:, :, i: i + self.patch]     # 公式(5) 时间维度交互

            res = tmp

            # [batch_size, feature_num, patch]
            if self.alpha > 0.0:
                tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
                tmp = torch.transpose(tmp, 1, 2)
                # [batch_size, patch, feature_num]
                bs, P, C = tmp.shape
                gnn_out = []
                head_dim = self.ff_dim // self.heads
                for p in range(P):
                    h = tmp[:, p, :].unsqueeze(-1)
                    h = self.channel_embed(h)
                    for _ in range(self.n_hop):
                        q = self.q_proj(h).view(bs, C, self.heads, head_dim).transpose(1, 2)
                        k = self.k_proj(h).view(bs, C, self.heads, head_dim).transpose(1, 2)
                        v = self.v_proj(h).view(bs, C, self.heads, head_dim).transpose(1, 2)
                        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                        if self.top_k < C:
                            topk_val, topk_idx = torch.topk(attn, self.top_k, dim=-1)
                            mask = torch.zeros_like(attn)
                            mask.scatter_(-1, topk_idx, 1.0)
                            attn = attn.masked_fill(mask == 0, -1e9)
                        attn = torch.softmax(attn, dim=-1)
                        h = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, C, self.ff_dim)
                    z_new = self.msg_mlp(h).squeeze(-1)
                    gnn_out.append(z_new.unsqueeze(1))
                tmp = torch.cat(gnn_out, dim=1).transpose(1, 2)
            output[:, :, i: i + self.patch] = res + self.alpha * tmp       # 公式（6） 通道维度交互（如果alpha等于0则没有加通道维度交互

        # [batch_size, feature_num, seq_len]
        return output

