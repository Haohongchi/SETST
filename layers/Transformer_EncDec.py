import torch
import torch.nn as nn
import torch.nn.functional as F
#assa 注意力模块
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from einops import rearrange
import math
import warnings
warnings.filterwarnings('ignore')

class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            attn = attn + mask.unsqueeze(1).unsqueeze(0)

        attn0 = self.softmax(attn)  # 密集注意力
        attn1 = self.relu(attn) ** 2  # 稀疏注意力

        # 计算自适应权重
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))

        # 融合稀疏和密集注意力
        attn = attn0 * w1 + attn1 * w2
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            attn = attn + mask.unsqueeze(1).unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x

class ASSA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., drop=0., norm_layer=nn.LayerNorm, sparseAtt=False):
        super().__init__()

        self.sparseAtt = sparseAtt
        self.dim = dim
        self.num_heads = num_heads

        # 使用稀疏或密集自注意力
        if self.sparseAtt:
            self.attn = WindowAttention_sparse(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
            )
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
            )

    def forward(self, x):
        B, C, L = x.shape  # B: batch size, C: number of variables, L: sequence length


        # 计算注意力
        x = self.attn(x)

        return x


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class GateFusion(nn.Module):
    def __init__(self, d_model,dropout):
        super(GateFusion, self).__init__()
        # 门控层
        self.gate_low = nn.Linear(d_model, d_model)
        self.gate_high = nn.Linear(d_model, d_model)

        # 线性变换层（可选）
        self.linear_low = nn.Linear(d_model, d_model)
        self.linear_high = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, y_low, y_high):
        # y_low 和 y_high 形状: [B, C, T]

        # 计算门控系数
        gate_low = torch.sigmoid(self.gate_low(y_low))  # 形状: [B, C, T]
        gate_high = torch.sigmoid(self.gate_high(y_high))  # 形状: [B, C, T]

        # 应用线性变换（可选）
        # y_low_transformed = self.linear_low(y_low)  # 形状: [B, C, T]
        # y_high_transformed = self.linear_high(y_high)  # 形状: [B, C, T]

        # y_low_transformed = self.dropout(y_low_transformed)
        # y_high_transformed = self.dropout(y_high_transformed)
        y_low_transformed = self.dropout(y_low)
        y_high_transformed = self.dropout(y_high)

        # 加权融合
        y_low_fused = gate_low * y_low_transformed + (1 - gate_low) * y_high_transformed
        y_high_fused = gate_high * y_high_transformed + (1 - gate_high) * y_low_transformed

        return y_low_fused, y_high_fused
#
# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1_low = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2_low = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.conv1_high = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2_high = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#
#         self.conv1_lo = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2_lo = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.conv1_hig = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2_hig = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         self.cross_guidance = GateFusion(d_model,dropout)
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x = x + self.dropout(new_x)
#
#         y = x = self.norm1(x)  #B    c   T
#
#         # 对 y 进行傅里叶变换
#         y_fft = torch.fft.fft(y, dim=-1)
#
#         # 计算低频和高频的边界
#         third_freq = y_fft.size(-1) // 3
#
#         # 初始化低频和高频部分
#         low_freq = torch.zeros_like(y_fft)
#         high_freq = torch.zeros_like(y_fft)
#
#         # 前三分之一频谱作为低频部分
#         low_freq[..., :third_freq] = y_fft[..., :third_freq]
#
#         # 后三分之一频谱作为高频部分
#         high_freq[..., -third_freq:] = y_fft[..., -third_freq:]
#
#         # 对低频和高频部分进行逆傅里叶变换
#         y_low = torch.fft.ifft(low_freq, dim=-1).real
#         y_high = torch.fft.ifft(high_freq, dim=-1).real
#
#         # 可以根据需要选择将低频和高频部分组合或分别处理
#         # y = y_low + y_high
#
#         y_low, y_high = self.cross_guidance(y_low, y_high)
#
#         y_low = self.dropout(self.activation(self.conv1_low(y_low.transpose(-1, 1))))
#         y_low = self.dropout(self.conv2_low(y_low).transpose(-1, 1))
#
#         y_high = self.dropout(self.activation(self.conv1_high(y_high.transpose(-1, 1))))
#         y_high = self.dropout(self.conv2_high(y_high).transpose(-1, 1))
#
#         # y_low, y_high = self.cross_guidance(y_low, y_high)
#         #
#         # y_low = self.dropout(self.activation(self.conv1_lo(y_low.transpose(-1, 1))))
#         # y_low = self.dropout(self.conv2_lo(y_low).transpose(-1, 1))
#         #
#         # y_high = self.dropout(self.activation(self.conv1_hig(y_high.transpose(-1, 1))))
#         # y_high = self.dropout(self.conv2_hig(y_high).transpose(-1, 1))
#
#         y = y_low + y_high
#
#         return self.norm2(x + y), attn

# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x = x + self.dropout(new_x)
#
#         y = x = self.norm1(x)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#
#         return self.norm2(x + y), attn

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.conv1_low = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_low = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv1_high = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_high = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.conv1_lo = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_lo = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv1_hig = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_hig = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.cross_guidance = GateFusion(d_model, dropout)

        # 线性层将输入转换为 Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_weights=None

        # 线性变换得到 Q, K, V
        Q = self.q_linear(x)  # (B, C, d_model)
        K = self.k_linear(x)  # (B, C, d_model)
        V = self.v_linear(x)  # (B, C, d_model)

        # 计算相似度矩阵
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, C, C)


        # 使用 ReLU 激活和平方操作替代 Softmax
        attn_scores = F.relu(attn_scores) ** 2

        # 对相似度矩阵进行归一化，使每个查询的权重之和为 1
        attn_weights = attn_scores / (attn_scores.sum(dim=-1, keepdim=True) + 1e-9)  # 避免除以零 (B, C, C)

        # 加权求和值
        new_x = torch.matmul(attn_weights, V)  # (B, C, d_model)

        # 残差连接 + LayerNorm
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        y_fft = torch.fft.fft(y, dim=-1)



        low_freq_end =  (y_fft.size(-1)  // 2) //  4



        # 初始化低频和高频部分
        low_freq = torch.zeros_like(y_fft)
        high_freq = torch.zeros_like(y_fft)



        low_freq[..., :low_freq_end] = y_fft[..., :low_freq_end]

        # 保留负频率部分的前 low_freq_end 个元素作为低频
        low_freq[..., -low_freq_end:] = y_fft[..., -low_freq_end:]

        # 保留正频率部分剩余的部分作为高频
        high_freq[..., low_freq_end:y_fft.size(-1) // 2] = y_fft[..., low_freq_end:y_fft.size(-1) // 2]

        # 保留负频率部分剩余的部分作为高频
        high_freq[..., y_fft.size(-1) // 2:-low_freq_end] = y_fft[..., y_fft.size(-1) // 2:-low_freq_end]



        # 对低频和高频部分进行逆傅里叶变换
        y_low = torch.fft.ifft(low_freq, dim=-1).real
        y_high = torch.fft.ifft(high_freq, dim=-1).real

        # 低频和高频部分的卷积处理
        y_low = self.dropout(self.activation(self.conv1_low(y_low.transpose(-1, 1))))
        y_low = self.dropout(self.conv2_low(y_low).transpose(-1, 1))

        y_high = self.dropout(self.activation(self.conv1_high(y_high.transpose(-1, 1))))
        y_high = self.dropout(self.conv2_high(y_high).transpose(-1, 1))

        # Cross-guidance 操作
        y_low, y_high = self.cross_guidance(y_low, y_high)

        y_low = self.dropout(self.activation(self.conv1_lo(y_low.transpose(-1, 1))))
        y_low = self.dropout(self.conv2_lo(y_low).transpose(-1, 1))

        y_high = self.dropout(self.activation(self.conv1_hig(y_high.transpose(-1, 1))))
        y_high = self.dropout(self.conv2_hig(y_high).transpose(-1, 1))

        # 低频和高频组合
        y = y_low + y_high

        return self.norm2(x + y), attn_weights





#
# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#         # 线性层将输入转换为 Q, K, V
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#
#
#         # 线性变换得到 Q, K, V
#         Q = self.q_linear(x)  # (B, C, d_model)
#         K = self.k_linear(x)  # (B, C, d_model)
#         V = self.v_linear(x)  # (B, C, d_model)
#
#         # 计算相似度矩阵
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, C, C)
#
#         # 使用 ReLU 激活和平方操作替代 Softmax
#         attn_scores = F.relu(attn_scores) ** 2
#
#         # 对相似度矩阵进行归一化，使每个查询的权重之和为 1
#         attn_weights = attn_scores / (attn_scores.sum(dim=-1, keepdim=True) + 1e-9)  # 避免除以零 (B, C, C)
#
#         # 加权求和值
#         new_x = torch.matmul(attn_weights, V)  # (B, C, d_model)
#
#         # 残差连接 + LayerNorm
#         x = x + self.dropout(new_x)
#         y = x = self.norm1(x)
#
#         # 前馈网络部分
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#
#         return self.norm2(x + y), attn_weights
# class Encoder(nn.Module):
#     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
#         self.norm = norm_layer
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         # x [B, L, D]
#         attns = []
#         if self.conv_layers is not None:
#             for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
#                 delta = delta if i == 0 else None
#                 x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
#                 x = conv_layer(x)
#                 attns.append(attn)
#             x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
#             attns.append(attn)
#         else:
#             # for attn_layer in self.attn_layers:
#             #     x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
#             #     attns.append(attn)
#             for attn_layer in self.attn_layers:
#                     x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
#                     attns.append(attn)
#
#         if self.norm is not None:
#             x = self.norm(x)
#
#         return x, attns

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1_low = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2_low = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.conv1_high = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2_high = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#
#         self.conv1_lo = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2_lo = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.conv1_hig = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2_hig = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         self.cross_guidance = GateFusion(d_model,dropout)
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x = x + self.dropout(new_x)
#
#         y = x = self.norm1(x)  #B    c   T
#
#         # 对 y 进行傅里叶变换
#         y_fft = torch.fft.fft(y, dim=-1)
#
#         # 计算低频和高频的边界
#         third_freq = y_fft.size(-1) // 4
#
#         # 初始化低频和高频部分
#         low_freq = torch.zeros_like(y_fft)
#         high_freq = torch.zeros_like(y_fft)
#
#         # 前三分之一频谱作为低频部分
#         low_freq[..., :third_freq] = y_fft[..., :third_freq]
#
#         # 后三分之一频谱作为高频部分
#         high_freq[..., -third_freq:] = y_fft[..., -third_freq:]
#
#         # 对低频和高频部分进行逆傅里叶变换
#         y_low = torch.fft.ifft(low_freq, dim=-1).real
#         y_high = torch.fft.ifft(high_freq, dim=-1).real
#
#         # 可以根据需要选择将低频和高频部分组合或分别处理
#         # y = y_low + y_high
#
#         # y_low, y_high = self.cross_guidance(y_low, y_high)
#
#         y_low = self.dropout(self.activation(self.conv1_low(y_low.transpose(-1, 1))))
#         y_low = self.dropout(self.conv2_low(y_low).transpose(-1, 1))
#
#         y_high = self.dropout(self.activation(self.conv1_high(y_high.transpose(-1, 1))))
#         y_high = self.dropout(self.conv2_high(y_high).transpose(-1, 1))
#
#         y_low, y_high = self.cross_guidance(y_low, y_high)
#
#         y_low = self.dropout(self.activation(self.conv1_lo(y_low.transpose(-1, 1))))
#         y_low = self.dropout(self.conv2_lo(y_low).transpose(-1, 1))
#
#         y_high = self.dropout(self.activation(self.conv1_hig(y_high.transpose(-1, 1))))
#         y_high = self.dropout(self.conv2_hig(y_high).transpose(-1, 1))
#
#         y = y_low + y_high
#
#         return self.norm2(x + y), attn

# class EncoderLayer1(nn.Module):
#     def __init__(self, enc_in, d_model, d_ff, dropout=0.1, activation="relu"):
#         super(EncoderLayer1, self).__init__()
#         d_ff = d_ff or 4 * d_model
#
#         self.vconv_1 = nn.Conv1d(in_channels=enc_in+4, out_channels=d_ff, kernel_size=1)
#         self.vconv_2 = nn.Conv1d(in_channels=d_ff, out_channels=enc_in+4, kernel_size=1)
#
#
#
#         self.conv1_low = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
#         self.conv2_low = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
#         self.conv1_high = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
#         self.conv2_high = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
#
#         self.conv1_lo = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
#         self.conv2_lo = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
#         self.conv1_hig = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
#         self.conv2_hig = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
#
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         self.cross_guidance = GateFusion(d_model,dropout)
#
#     def forward(self, x):
#         new_x = self.dropout(self.activation(self.vconv_1(x)))
#         new_x = self.dropout(self.vconv_2(new_x))
#
#         x = x + new_x
#
#         y = x = self.norm1(x)  #B    c   T
#
#         # 对 y 进行傅里叶变换
#         y_fft = torch.fft.fft(y, dim=-1)
#
#         # 计算低频和高频的边界
#         third_freq = y_fft.size(-1) // 3
#
#         # 初始化低频和高频部分
#         low_freq = torch.zeros_like(y_fft)
#         high_freq = torch.zeros_like(y_fft)
#
#         # 前三分之一频谱作为低频部分
#         low_freq[..., :third_freq] = y_fft[..., :third_freq]
#
#         # 后三分之一频谱作为高频部分
#         high_freq[..., -third_freq:] = y_fft[..., -third_freq:]
#
#         # 对低频和高频部分进行逆傅里叶变换
#         y_low = torch.fft.ifft(low_freq, dim=-1).real
#         y_high = torch.fft.ifft(high_freq, dim=-1).real
#
#         # 可以根据需要选择将低频和高频部分组合或分别处理
#         # y = y_low + y_high
#
#         # y_low, y_high = self.cross_guidance(y_low, y_high)
#
#         y_low = self.dropout(self.activation(self.conv1_low(y_low.transpose(-1, 1))))
#         y_low = self.dropout(self.conv2_low(y_low).transpose(-1, 1))
#
#         y_high = self.dropout(self.activation(self.conv1_high(y_high.transpose(-1, 1))))
#         y_high = self.dropout(self.conv2_high(y_high).transpose(-1, 1))
#
#         y_low, y_high = self.cross_guidance(y_low, y_high)
#
#         y_low = self.dropout(self.activation(self.conv1_lo(y_low.transpose(-1, 1))))
#         y_low = self.dropout(self.conv2_lo(y_low).transpose(-1, 1))
#
#         y_high = self.dropout(self.activation(self.conv1_hig(y_high.transpose(-1, 1))))
#         y_high = self.dropout(self.conv2_hig(y_high).transpose(-1, 1))
#
#         y = y_low + y_high
#
#         return self.norm2(x + y)


class EncoderLayer1(nn.Module):

    def __init__(self, enc_in, d_model, d_ff=None, dropout=0.1, activation="relu", sparse_attn=True):
        super(EncoderLayer1, self).__init__()
        d_ff = d_ff or 4 * d_model

        # 使用 ASSA 代替原来的 attention 模块
        self.attention = ASSA(dim=d_model, num_heads=8, sparseAtt=sparse_attn)

        # 保持原始的卷积层和前馈部分
        self.conv1_low = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_low = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv1_high = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_high = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.conv1_lo = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_lo = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv1_hig = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_hig = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.cross_guidance = GateFusion(d_model, dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 使用 ASSA 模块计算自注意力
        new_x = self.attention(x)

        # 残差连接和 Dropout
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # 对 y 进行傅里叶变换
        y_fft = torch.fft.fft(y, dim=-1)

        # 计算低频和高频的边界
        third_freq = y_fft.size(-1) // 4

        # 初始化低频和高频部分
        low_freq = torch.zeros_like(y_fft)
        high_freq = torch.zeros_like(y_fft)

        # 前三分之一频谱作为低频部分
        low_freq[..., :third_freq] = y_fft[..., :third_freq]

        # 后三分之一频谱作为高频部分
        high_freq[..., -third_freq:] = y_fft[..., -third_freq:]

        # 对低频和高频部分进行逆傅里叶变换
        y_low = torch.fft.ifft(low_freq, dim=-1).real
        y_high = torch.fft.ifft(high_freq, dim=-1).real

        # 低频和高频部分的卷积处理
        y_low = self.dropout(self.activation(self.conv1_low(y_low.transpose(-1, 1))))
        y_low = self.dropout(self.conv2_low(y_low).transpose(-1, 1))

        y_high = self.dropout(self.activation(self.conv1_high(y_high.transpose(-1, 1))))
        y_high = self.dropout(self.conv2_high(y_high).transpose(-1, 1))

        # Cross-guidance 操作
        y_low, y_high = self.cross_guidance(y_low, y_high)

        y_low = self.dropout(self.activation(self.conv1_lo(y_low.transpose(-1, 1))))
        y_low = self.dropout(self.conv2_lo(y_low).transpose(-1, 1))

        y_high = self.dropout(self.activation(self.conv1_hig(y_high.transpose(-1, 1))))
        y_high = self.dropout(self.conv2_hig(y_high).transpose(-1, 1))

        # 低频和高频组合
        y = y_low + y_high

        return self.norm2(x + y)
