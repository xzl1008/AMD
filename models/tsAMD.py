import torch
import torch.nn as nn

from models.common import RevIN
from models.common import DDI
from models.common import MDM
from models.tsmoe import AMS


class AMD(nn.Module):
    """Implementation of AMD."""

    def __init__(self, input_shape, pred_len, n_block, dropout, patch, k, c, alpha, target_slice, norm=True, layernorm=True):
        """AMD模型的初始化函数
            Parameters
            ----------
            input_shape : tuple
                输入序列的形状 (seq_len, feature_num)，用于确定序列长度和特征维度。
                在MDM、DDI与AMS模块中都会使用到这个参数。

            pred_len : int
                需要预测的序列长度，用于AMS模块生成对应长度的输出。

            n_block : int
                DDI模块堆叠的数量，模型会构造n_block个DDI层来提取时序依赖。

            dropout : float
                Dropout概率值，取值范围[0,1)。
                用于DDI模块的fc_block和AMS模块的专家网络中。

            patch : int
                DDI模块中的时间片长度(历史窗口大小)，
                用于内部的聚合和归一化操作。

            k : int
                MDM模块中多尺度池化的层数。
                与参数c一起决定不同尺度池化窗口的数量和大小。

            c : int
                MDM模块中池化层尺度的倍数。
                通过公式[c**i for i in range(k, 0, -1)]计算不同池化窗口的尺度。

            alpha : float
                DDI模块中跨特征交互强度的控制系数。
                当alpha > 0时，会额外启用特征维度上的fc_block，并用alpha调节输出强度。

            target_slice : slice
                目标特征的切片位置，支持Python切片语法。
                用途：
                1. 在反归一化时只对目标维度操作
                2. 在最终输出时取出目标维度

            norm : bool, optional
                是否启用RevIN可逆归一化，默认为True。
                当为True时，在前向传播前后会执行可逆归一化处理。
                RevIN 的作用是对每个时间序列样本进行可逆的实例归一化（Instance Normalization），
                以减轻不同序列之间的分布差异，并在模型预测后恢复原始尺度。
                这样既能减少训练和测试之间的分布偏移，又不会改变预测值的物理意义。

            layernorm : bool, optional
                控制MDM和DDI内部是否使用BatchNorm1d，默认为True。
                当为False时，模块内部不会使用Layer Normalization。
                Layer Normalization（简称 LayerNorm）是一种在神经网络训练中常用的归一化技术。
                它在每个样本内部，对所有特征维度（或隐藏单元）进行均值和方差的归一化。与 BatchNorm 需要在批次维度上统计均值方差不同，
                LayerNorm 在单个样本内部归一化，对小批量甚至单样本也能正常工作。

            Attributes
            ----------
            pastmixing : MDM
                多尺度分解模块的实例

            fc_blocks : nn.ModuleList
                包含n_block个DDI模块的列表

            moe : AMS
                自适应混合专家模块的实例
        """
        super(AMD, self).__init__()

        self.target_slice = target_slice
        self.norm = norm

        if self.norm:
            self.rev_norm = RevIN(input_shape[-1])

        self.pastmixing = MDM(input_shape, k=k, c=c, layernorm=layernorm)

        self.fc_blocks = nn.ModuleList([DDI(input_shape, dropout=dropout, patch=patch, alpha=alpha, layernorm=layernorm)
                                        for _ in range(n_block)])

        self.moe = AMS(input_shape, pred_len, ff_dim=2048, dropout=dropout, num_experts=8, top_k=2)

    def forward(self, x):
        # [batch_size, seq_len, feature_num]

        # layer norm
        if self.norm:
            x = self.rev_norm(x, 'norm')
        # [batch_size, seq_len, feature_num]

        # [batch_size, seq_len, feature_num]
        x = torch.transpose(x, 1, 2)
        # [batch_size, feature_num, seq_len]

        time_embedding = self.pastmixing(x)

        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # MOE
        x, moe_loss = self.moe(x, time_embedding)  # seq_len -> pred_len

        # [batch_size, feature_num, pred_len]
        x = torch.transpose(x, 1, 2)
        # [batch_size, pred_len, feature_num]

        if self.norm:
            x = self.rev_norm(x, 'denorm', self.target_slice)
        # [batch_size, pred_len, feature_num]

        if self.target_slice:
            x = x[:, :, self.target_slice]

        return x, moe_loss