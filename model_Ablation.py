import torch.nn as nn
import torch
from torch.nn.utils import parametrizations
import numpy as np

from modules import ConvLayer
from tcn import TemporalConvNetSENet


# 定义裁剪模块
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 定义 TCN 卷积+残差 模块
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5):
        super(TemporalBlock, self).__init__()
        """
        params: 构成TCN的核心Block, 原作者在图中成为Residual block, 是因为它存在残差连接.
        但注意, 这个模块包含了2个Conv1d.

        n_inputs         : 输入通道数或者特征数
        n_outputs        : 输出通道数或者特征数
        kernel_size      : 卷积核大小
        stride           : 步长, 在TCN固定为1
        dilation         : 膨胀系数. 与这个Residual block(或者说, 隐藏层)所在的层数有关系. 
                            例如, 如果这个Residual block在第1层, dilation = 2**0 = 1;
                                    如果这个Residual block在第2层, dilation = 2**1 = 2;
                                    如果这个Residual block在第3层, dilation = 2**2 = 4;
                                    如果这个Residual block在第4层, dilation = 2**3 = 8 ......
        padding          : 填充系数. 与kernel_size和dilation有关
        dropout          : drop_out比率
        """
        # 第一层 卷积
        self.conv1 = parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                            stride=stride, padding=padding, dilation=dilation))
        # 因为 padding 的时候, 在序列的左边和右边都有填充, 所以要裁剪
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层 卷积
        self.conv2 = parametrizations.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 1×1的卷积. 只有在进入Residual block的通道数与出Residual block的通道数不一样时使用.
        # 一般都会不一样, 除非num_channels这个里面的数, 与num_inputs相等. 例如[5,5,5], 并且num_inputs也是5
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # 进行下采样

        # 在整个Residual block中有非线性的激活
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = out + res
        out = self.relu(out)
        return out


# 通道注意力机制
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化  x: torch.Size([256, 128, 121])
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        # Squeeze阶段：通过全连接层降维
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # Excitation阶段：通过Sigmoid函数生成权重
        out = self.sigmoid(out)  # out: torch.Size([256, 128])
        out = out.view(out.size(0), out.size(1), -1)
        # 乘以权重，实现通道的重新加权
        out = x * out
        return out


# 定义 Global-Attention 注意力机制
class GlobalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        repeated_hidden = hidden.unsqueeze(1).repeat(1, max_len, 2)
        energy = torch.tanh(self.attn(torch.cat((repeated_hidden, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        context_vector = (encoder_outputs * attention_weights.unsqueeze(2)).sum(dim=1)
        return context_vector


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.hid_dim = hid_dim
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """重建模型
    :param window_size: 输入序列的长度
    :param in_dim: 输入特征的数量
    :param n_layers: RNN 的层数
    :param hid_dim: RNN 的隐藏层大小
    :param out_dim: 输出特征的数量
    :param dropout: dropout 率
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)
        # 初始值为 0，但它会在训练过程中根据梯度更新
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, x):
        # x 是 GRU 层的最后一个隐藏状态
        h_end = x
        # 将最后一个隐藏状态 h_end 沿着时间步长重复，并调整形状
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        # 通过解码器
        decoder_out = self.decoder(h_end_rep)

        #decoder_out = self.norm(decoder_out + h_end_rep)  # 残差连接加归一化
        decoder_out = decoder_out + self.resweight * decoder_out

        # 通过全连接层
        out = self.fc(decoder_out)

        return out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 添加一个新的全连接层
        #self.fc = nn.Linear(num_channels[-1] + hidden_layer_sizes[-1] * 2, out_dim)

    def forward(self, x):
        #print("预测模块",len(self.layers))
        for i in range(len(self.layers)):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)

        # 使用新的全连接层
        # x = self.fc(x)
        return x


# 定义 TCNSENet-BiGRUGlobalAttention  并行网络预测模型
class TCNSENetBiGRUGlobalAttention(nn.Module):
    def __init__(self, window_size, batch_size, input_dim, output_dim, num_channels, kernel_size,
                 hidden_layer_sizes, attention_dim,
                 dropout, gru_hid_dim, recon_hid_dim, recon_n_layers):
        super(TCNSENetBiGRUGlobalAttention, self).__init__()
        """
        params:
        batch_size         : 批次量大小
        input_dim          : 输入数据的维度
        output_dim         : 输出维度
        num_channels       : 每个TemporalBlock中的输出通道数 
                                例如[5,12,3], 代表有3个block, 
                                block1的输出channel数量为5; 
                                block2的输出channel数量为12;
                                block3的输出channel数量为3.
        kernel_size        : 卷积核大小
        hidden_layer_sizes : BiGRU隐藏层的数目和维度
        attention_dim      : 全局注意力维度
        dropout            : drop_out比率
        """
        # 参数
        self.batch_size = batch_size
        self.num_c = num_channels[-1] + hidden_layer_sizes[- 1] * 2
        fc_layers = 1
        # TCN 时序空间特征 参数
        self.conv = ConvLayer(input_dim, kernel_size)
        self.TCNnetwork = TemporalConvNetSENet(input_dim, num_channels, kernel_size=kernel_size)
        # 序列平局池化  为什么加这个进去，输入 [[64, 64, 7] ->[[64, 64, 1] 序列长度 做一个平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # BiGRU参数
        self.num_layers = len(hidden_layer_sizes)  # bigru层数
        self.bigru_layers = nn.ModuleList()  # 用于保存BiGRU层的列表
        # 定义第一层BiGRU
        self.bigru_layers.append(nn.GRU(input_dim, hidden_layer_sizes[0], batch_first=True, bidirectional=True))
        # 定义后续的BiGRU层
        for i in range(1, self.num_layers):
            self.bigru_layers.append(
                nn.GRU(hidden_layer_sizes[i - 1] * 2, hidden_layer_sizes[i], batch_first=True, bidirectional=True))

        # 定义 全局注意力层
        self.globalAttention = GlobalAttention(attention_dim * 2)  # 双向GRU 维度 *2

        self.fc = nn.Linear(num_channels[-1]*2, output_dim)

        self.recon_model = ReconstructionModel(window_size, num_channels[-1]*2, num_channels[-1]*2, output_dim,
                                               recon_n_layers,
                                               dropout)

    def forward(self, input_seq):

        input_seq = self.conv(input_seq)
        # 分支一：
        # 时序空间特征 提取
        # TCN-SENet 输入 （batch_size, channels, length）
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # 调换维度[B, L, D] --> [B, D, L]
        tcn_input = input_seq.permute(0, 2, 1)

        tcnSENet_features = self.TCNnetwork(tcn_input)
        # print("tcnSENet_features:",tcnSENet_features.size())   # tcnSENet_features.size() ([256,64,90])
        # 自适应平均池化
        tcnSENet_features = self.avgpool(tcnSENet_features)  # torch.Size([64, 64, 1])
        # print("池化tcnSENet_features:", tcnSENet_features.size())
        # #池化tcnSENet_features: torch.Size([256, 64, 1])

        # 平铺
        # tcnSENet_features = tcnSENet_features.reshape(self.batch_size, -1)
        #平铺tcnSENet_features: torch.Size([256, 64])
        tcnSENet_features = tcnSENet_features.squeeze(2)  # 平铺tcnSENet_features: torch.Size([256, 64])

        # 分支二：
        # 时域特征 送入BiGRU
        # 输入形状，适应网络输入[batch, seq_length, H_in]
        # print("input_seq: ", input_seq.shape)
        bigru_out = input_seq
        # hidden 获取隐藏层数据
        for bigru in self.bigru_layers:
            bigru_out, hidden = bigru(bigru_out)  # 进行一次BiGRU层的前向传播   (b, l, w)

        # 送入全局注意力层
        gatt_features = self.globalAttention(hidden[-1], bigru_out)  # torch.Size([256, 128])

        # 并行融合特征
        # print("填充后的gatt_features: ", gatt_features.shape)   (240,128)
        # combined_features = torch.cat((tcnSENet_features, gatt_features), dim=1)  # torch.Size([32, 64 + 128])
        combined_features = gatt_features
        recon = self.recon_model(combined_features)
        predict = self.fc(combined_features)  # 输出维度为[batch_size, output_dim]
        # recon = recon.squeeze()
        # recon = recon.mean(dim=1, keepdim=True)
        return predict, recon
