import torch
import torch.nn as nn


class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, noise_epsilon=1e-5):
        super(TopKGating, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)    #  创建一个线性层 self.gate，将时间信息映射到每个专家的得分。
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        self.num_experts = num_experts
        self.w_noise = nn.Parameter(torch.zeros(num_experts, num_experts), requires_grad=True)   # 初始化噪声权重矩阵
        self.softplus = nn.Softplus()
        """
            Softplus 是一种光滑的激活函数，可以看作是 ReLU 的平滑版本。它的输出始终为正，且在输入较大时近似线性，
            在输入较小时则类似于指数增长，因而常被用于需要生成正值但又希望梯度连续的场景。
            在训练阶段，门控分支会用它把噪声的标准差 raw_noise_stddev 映射为正数，从而生成高斯噪声。
            这里 softplus 确保 noise_stddev 为正，随后加入到门控得分中形成 noisy_logits。如此一来可以在训练时引入可学习的抖动，
            使模型更好地探索不同专家组合。整体来说，softplus 兼具“输出非负”和“梯度平滑”两个特性，
            因此常用于需要生成正数且希望在零点附近保持可导的场景。
        """
        self.softmax = nn.Softmax(1)
        """
            参数 1 表示沿输入张量的第 1 维（即第二个维度）计算。因此当 x 的形状为 [batch_size, num_experts] 时，
            nn.Softmax(1) 会在每一行上计算软最大值，使得每个样本对应的所有专家得分之和为 1，形成概率分布。
            这在门控网络中用于表示各专家被选择的权重。
        """

    def decompostion_tp(self, x, alpha=10):
        # x [batch_size, seq_len]
        output = torch.zeros_like(x)
        # [batch_size]
        # 先找出每个样本中第 num_experts - top_k + 1 大的位置（即剩余专家中最大的阈值）。
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1)
        # [batch_size, num_expert]
        kth_largest_mat = kth_largest_val.unsqueeze(1).expand(-1, self.num_experts)
        mask = x < kth_largest_mat
        x = self.softmax(x)
        output[mask] = alpha * torch.log(x[mask] + 1)    # 根据掩码将不重要专家的得分经过 log 调制
        output[~mask] = alpha * (torch.exp(x[~mask]) - 1)    # 将重要专家的得分用 exp 放大
        # Ablation Spare MoE
        # output[mask] = 0
        # [batch_size, seq_len]
        return output

    def forward(self, x):
        # [batch_size, seq_len]

        x = self.gate(x)
        clean_logits = x    # 通过 self.gate 得到每个专家的初始得分 clean_logits
        # [batch_size, num_experts]

        if self.training:
            """
                训练时引入可学习的高斯噪声：得分先与 self.w_noise 相乘得到标准差，
                再通过 Softplus + noise_epsilon 保证数值稳定，之后加入噪声形成 noisy_logits。推理阶段直接使用 clean_logits。
                ------------------
                w_noise 是学习得到的矩阵参数，用于控制各专家得分的噪声幅度。
                在训练阶段加入噪声，能够在梯度更新时让模型探索更多的专家组合，防止门控层过早收敛到固定的专家分配，从而实现更好的负载均衡；
                而在推理（eval）阶段则使用纯净的得分以获得稳定的输出。这样既保持了训练阶段的多样性，又保证了预测阶段的确定性。
            """
            raw_noise_stddev = x @ self.w_noise
            """
                x 和 self.w_noise 均为张量（矩阵），@ 运算符对它们进行矩阵乘法，得到噪声的标准差 raw_noise_stddev。
                因此，@ 的作用与 torch.matmul() 或 numpy.matmul() 等价。它可以简化书写且更符合线性代数表示习惯。
            """
            noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)   # 公式（8）
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.decompostion_tp(logits)     # 进一步调整得分
        gates = self.softmax(logits)       # 公式（7），做 softmax 得到最终的 gates（每个样本、每个专家的权重）。

        # random order
        # indices = torch.randperm(gates.size(0))
        # shuffled_gates = gates[indices]

        # average
        # value = 1.0 / x.shape[1]
        # gates = torch.full(x.shape, value, device=x.device)

        return gates


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.2):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class AMS(nn.Module):
    def __init__(self, input_shape, pred_len, ff_dim=2048, dropout=0.2, loss_coef=1.0, num_experts=4, top_k=2):
        """
        AMS模型的初始化函数
            Parameters
            ----------
            input_shape：输入序列的形状，格式为 (seq_len, feature_num)，其中 seq_len 表示历史序列长度，feature_num 表示特征数量。
            pred_len：预测的时间步长度，即模型最终要输出多少步的结果。
            ff_dim：每个 Expert 网络内部前馈层的隐藏维度，默认 2048。在构建 Expert 时作为 hidden_dim 传入。
            dropout：Expert 网络中 Dropout 的概率，默认 0.2。
            loss_coef：控制门控负载均衡损失的权重。在前向传播中，该系数乘以 cv_squared(importance) 累加到 loss 上，用于鼓励专家使用更加均衡。
            num_experts：专家网络数量，默认为 4。对应 self.num_experts，并在构建 nn.ModuleList 时生成相应个数的 Expert 模块。
            top_k：TopKGating 选择的专家个数。TopKGating 根据时间信息为每个批次选择 top_k 个专家参与预测，并保证 top_k <= num_experts。
        """
        super(AMS, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.num_experts = num_experts
        self.top_k = top_k
        self.pred_len = pred_len

        self.gating = TopKGating(input_shape[0], num_experts, top_k)

        self.experts = nn.ModuleList(
            [Expert(input_shape[0], pred_len, hidden_dim=ff_dim, dropout=dropout) for _ in range(num_experts)])
        self.loss_coef = loss_coef
        assert (self.top_k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, time_embedding):
        # [batch_size, feature_num, seq_len]
        batch_size = x.shape[0]
        feature_num = x.shape[1]
        # [batch_size, feature_num, seq_len] -> [feature_num, batch_size, seq_len]
        x = torch.transpose(x, 0, 1)
        # [batch_size, feature_num, seq_len] -> [feature_num, batch_size, seq_len]
        time_embedding = torch.transpose(time_embedding, 0, 1)
        """
            将输入x和时间嵌入time_embedding从 [batch, feature, seq] 转置为 [feature, batch, seq]，便于逐特征处理。
        """
        output = torch.zeros(feature_num, batch_size, self.pred_len).to(x.device)
        loss = 0

        for i in range(feature_num):   # 遍历每个特征
            input = x[i]    # 取出第i个feature对应的历史输入。
            time_info = time_embedding[i]    # 取出第i个feature的时间信息。
            # x[i]  [batch_size, seq_len]
            gates = self.gating(time_info)     # 计算 gating 权重。

            # expert_outputs [batch_size, num_experts, pred_len]
            # 更正：expert_outputs [num_experts, batch_size, pred_len]
            expert_outputs = torch.zeros(self.num_experts, batch_size, self.pred_len).to(x.device)

            for j in range(self.num_experts):     # 收集所有专家对该特征的预测结果。
                expert_outputs[j, :, :] = self.experts[j](input)
            # expert_outputs [batch_size, num_experts, pred_len]
            expert_outputs = torch.transpose(expert_outputs, 0, 1)
            # gates [batch_size, seq_len] -> [batch_size, num_experts, pred_len]
            gates = gates.unsqueeze(-1).expand(-1, -1, self.pred_len)      # 扩展 gating 权重到预测长度
            """
                unsqueeze(dim)：在指定维度插入一个新的维度（大小为 1）。dim=-1 表示最后一个维度。
                expand(*sizes)：将张量的特定维度扩展为更大的尺寸。-1 表示保持当前维度不变。self.pred_len 是预测长度，即需要将最后一维扩展为该长度。
            """
            # batch_output [batch_size, pred_len]
            batch_output = (gates * expert_outputs).sum(1)    # 在维度1，即num_experts维度求和。公式（9）多预测器加权输出。
                                                              # gating与专家结果相乘后求和得到最终预测
            output[i, :, :] = batch_output

            importance = gates.sum(0)     # 计算各专家的重要度
            loss += self.loss_coef * self.cv_squared(importance)    # 累积负载均衡损失。

        # [feature_num, batch_size, seq_len]
        output = torch.transpose(output, 0, 1)
        # [batch_size, feature_num, seq_len]

        return output, loss     # 这个loss为公式（11）中的Loss Selector


class AMSE(nn.Module):     # 在总损失中加入熵正则的AMS模型
    def __init__(self, input_shape, pred_len, ff_dim=2048, dropout=0.2, loss_coef=1.0, entropy_coef=1.0,
                 num_experts=4, top_k=2, entropy_eps=1e-8):
        """
        AMS模型的初始化函数
            Parameters
            ----------
            input_shape：输入序列的形状，格式为 (seq_len, feature_num)，其中 seq_len 表示历史序列长度，feature_num 表示特征数量。
            pred_len：预测的时间步长度，即模型最终要输出多少步的结果。
            ff_dim：每个 Expert 网络内部前馈层的隐藏维度，默认 2048。在构建 Expert 时作为 hidden_dim 传入。
            dropout：Expert 网络中 Dropout 的概率，默认 0.2。
            loss_coef：控制门控负载均衡损失的权重。在前向传播中，该系数乘以 cv_squared(importance) 累加到 loss 上，用于鼓励专家使用更加均衡。
            entropy_coef：熵正则项的权重 λ3，用于鼓励选择结果更稀疏明确。
            num_experts：专家网络数量，默认为 4。对应 self.num_experts，并在构建 nn.ModuleList 时生成相应个数的 Expert 模块。
            top_k：TopKGating 选择的专家个数。TopKGating 根据时间信息为每个批次选择 top_k 个专家参与预测，并保证 top_k <= num_experts。
        """
        super(AMSE, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.num_experts = num_experts
        self.top_k = top_k
        self.pred_len = pred_len

        self.gating = TopKGating(input_shape[0], num_experts, top_k)

        self.experts = nn.ModuleList(
            [Expert(input_shape[0], pred_len, hidden_dim=ff_dim, dropout=dropout) for _ in range(num_experts)])
        self.loss_coef = loss_coef
        self.entropy_coef = entropy_coef
        self.entropy_eps = entropy_eps
        assert (self.top_k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, time_embedding):
        # [batch_size, feature_num, seq_len]
        batch_size = x.shape[0]
        feature_num = x.shape[1]
        # [batch_size, feature_num, seq_len] -> [feature_num, batch_size, seq_len]
        x = torch.transpose(x, 0, 1)
        # [batch_size, feature_num, seq_len] -> [feature_num, batch_size, seq_len]
        time_embedding = torch.transpose(time_embedding, 0, 1)
        """
            将输入x和时间嵌入time_embedding从 [batch, feature, seq] 转置为 [feature, batch, seq]，便于逐特征处理。
        """
        output = torch.zeros(feature_num, batch_size, self.pred_len).to(x.device)
        selector_loss = 0
        entropy_loss = 0

        for i in range(feature_num):  # 遍历每个特征
            input = x[i]  # 取出第i个feature对应的历史输入。
            time_info = time_embedding[i]  # 取出第i个feature的时间信息。
            # x[i]  [batch_size, seq_len]
            gates = self.gating(time_info)  # 计算 gating 权重。

            #entropy_loss += (-gates * torch.log(gates + self.entropy_eps)).sum(dim=1).mean()  # 计算熵正则

            # expert_outputs [batch_size, num_experts, pred_len]
            # 更正：expert_outputs [num_experts, batch_size, pred_len]
            expert_outputs = torch.zeros(self.num_experts, batch_size, self.pred_len).to(x.device)

            for j in range(self.num_experts):  # 收集所有专家对该特征的预测结果。
                expert_outputs[j, :, :] = self.experts[j](input)
            # expert_outputs [batch_size, num_experts, pred_len]
            expert_outputs = torch.transpose(expert_outputs, 0, 1)
            # gates [batch_size, seq_len] -> [batch_size, num_experts, pred_len]
            gates = gates.unsqueeze(-1).expand(-1, -1, self.pred_len)  # 扩展 gating 权重到预测长度
            """
                unsqueeze(dim)：在指定维度插入一个新的维度（大小为 1）。dim=-1 表示最后一个维度。
                expand(*sizes)：将张量的特定维度扩展为更大的尺寸。-1 表示保持当前维度不变。self.pred_len 是预测长度，即需要将最后一维扩展为该长度。
            """
            # batch_output [batch_size, pred_len]
            batch_output = (gates * expert_outputs).sum(1)  # 在维度1，即num_experts维度求和。公式（9）多预测器加权输出。
            # gating与专家结果相乘后求和得到最终预测
            output[i, :, :] = batch_output

            importance = gates.sum(0)  # 计算各专家的重要度
            selector_loss += self.loss_coef * self.cv_squared(importance)  # 累积负载均衡损失。


            importance1 = gates.mean(0)
            entropy_loss += (-importance1 * torch.log(importance1 + self.entropy_eps)).mean()  # 计算熵正则

        # [feature_num, batch_size, seq_len]
        output = torch.transpose(output, 0, 1)
        # [batch_size, feature_num, seq_len]
        # total_loss = selector_loss + self.entropy_coef * entropy_loss
        return output, selector_loss, entropy_loss  # 这个loss为公式（11）中的Loss Selector，加了熵正则