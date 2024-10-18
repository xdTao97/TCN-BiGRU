import torch
import torch.nn as nn

from model import GlobalAttention
from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    # GRULayer,
    # Forecasting_Model,
    # ReconstructionModel,
    TCN_Model,
    SingleStageTCN,
    aggregation,
    MultiStageModel,
    BiGRU,
    multi_feature_GAT,
    multi_temporal_GAT
)


class MTAD_GAT(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    # :param gru_n_layers: number of layers in the GRU layer
    # :param gru_hid_dim: hidden dimension in the GRU layer
    # :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    # :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    # :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    # :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
            self,
            n_features,
            window_size,
            out_dim,
            conv_kernel_size=7,
            feat_gat_embed_dim=None,
            time_gat_embed_dim=None,
            use_gatv2=True,
            alpha=0.2,
            dropout=0.2,
            tcn_kernel_size=4,
            tcn_nhid=150,
            tcn_levels=4,
            num_heads = 2
    ):
        super(MTAD_GAT, self).__init__()
        hidden_layer_sizes = [32, 64, 128]
        #self.conv = ConvLayer(n_features, conv_kernel_size)
        #self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        #self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim,
        #                                          use_gatv2)
        #self.multi_feature_GAT = multi_feature_GAT(n_features, window_size, dropout, alpha, num_heads)

        #self.multi_temporal_GAT = multi_temporal_GAT(n_features, window_size, dropout, alpha, num_heads)
        num_channels = [tcn_nhid] * tcn_levels
        # num_channels = [100, 100, 100]

        self.tcn = TCN_Model(n_features, out_dim, num_channels=num_channels, kernel_size=tcn_kernel_size,
                              dropout=dropout)
        # 序列平局池化  为什么加这个进去，输入 [[64, 64, 7] ->[[64, 64, 1] 序列长度 做一个平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.bigru = BiGRU(n_features, hidden_layer_sizes)
        #self.aggregation = aggregation(window = window_size,channel=n_features)
        #self.SingleStageTCN = SingleStageTCN(3 * n_features, n_features, 4)
        self.globalAttention = GlobalAttention(hidden_layer_sizes[-1]*2)
        self.aggregation = aggregation(window=window_size, channel=n_features)

    def forward(self, input_seq):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        '''
        x = self.conv(x)
        h_feat = self.multi_feature_GAT(x)
        h_temp = self.multi_temporal_GAT(x)

        agg = self.aggregation(x, h_feat, h_temp)
        predictions = self.SingleStageTCN(agg)
        '''

        # 分支一：
        # 时序空间特征 提取
        # TCN-SENet 输入 （batch_size, channels, length）
        # 调换维度[B, L, D] --> [B, D, L]
        tcn_input = input_seq.permute(0, 2, 1)
        tcnSENet_features = self.tcn(tcn_input)
        # print(tcn_features.size())   # torch.Size([64, 64, 7])
        # 自适应平均池化
        tcnSENet_features = self.avgpool(tcnSENet_features)  # torch.Size([64, 64, 1])
        # 平铺
        tcnSENet_features = tcnSENet_features.reshape(self.batch_size, -1)  # torch.Size([64, 64])

        # 分之二：
        # 时域特征 送入BiGRU
        # 输入形状，适应网络输入[batch, seq_length, H_in]
        bigru_out = input_seq
        # hidden 获取隐藏层数据
        bigru_out, hidden = self.bigru(bigru_out)
        #送入全局注意力层
        gatt_features = self.globalAttention(hidden[-1], bigru_out)  # torch.Size([64, 128])
        # 并行融合特征
        combined_features = torch.cat((tcnSENet_features, gatt_features), dim=1)  # torch.Size([32, 64 + 128])
        predict = self.fc(combined_features)  # 输出维度为[batch_size, output_dim]


        return predict
