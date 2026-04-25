import math

import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from rms_norm import RMSNorm


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale          # 缩放因子（默认1/sqrt(dim)）
        self.mask_flag = mask_flag  # 是否启用因果掩码（防止未来信息泄漏）
        self.output_attention = output_attention  # 是否返回注意力权重
        self.dropout = nn.Dropout(attention_dropout)  # 注意力权重随机丢弃

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape  # queries: (Batch大小, 序列长度, 注意力头数, Query嵌入维度）
        _, S, _, D = values.shape   # keys: [B, S, H, E]（Value的维度D通常与E相同）
        _, S, _, D = values.shape   # Value: [B, S, H, D]

        #  计算缩放点积注意力分数
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # 输出形状 [B, H, L, S]

        # 掩码处理 生成下三角因果卷积
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)  # 可选的掩码矩阵（如因果掩码）

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 用注意力权重 A 对 values 加权求和，输出形状 [B, L, H, D]
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class DiffAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, depth=0, attention_dropout=0.1, embed_dim=512,
                 num_heads=8, num_kv_heads=None, output_attention=False):
        super(DiffAttention, self).__init__()
        self.scale = scale          # 缩放因子（默认1/sqrt(dim)）
        self.mask_flag = mask_flag  # 是否启用因果掩码（防止未来信息泄漏）
        self.output_attention = output_attention  # 是否返回注意力权重
        self.dropout = nn.Dropout(attention_dropout)  # 注意力权重随机丢弃

        self.num_heads = num_heads // 2
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else self.num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = embed_dim // self.num_heads // 2
        self.scaling = self.head_dim ** -0.5


        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, q, k, v, attn_mask, tau=None, delta=None):
        B, L, H, E = q.shape  # queries: (Batch大小, 序列长度, 注意力头数, Query嵌入维度）
        _, S, _, D = v.shape   # keys: [B, S, H, E]（Value的维度D通常与E相同）
        _, S, _, D = v.shape   # Value: [B, S, H, D]

        q = q.view(B, L, 2 * self.num_heads, self.head_dim)
        k = k.view(B, L, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(B, L, self.num_kv_heads, 2 * self.head_dim)

        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)

        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([L, L])
                    .float()
                    .fill_(float("-inf"))
                    .type_as(attn_weights),
                    1 + 0,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        # print(attn_weights.shape)
        attn_weights = attn_weights.view(B, self.num_heads, 2, L, L)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        # print(v.shape)

        attn = torch.matmul(attn_weights, v)
        # attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(B, L, H, E)
        V = attn

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Conv1d(  # 改为 Conv1d
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels * 2)  # 改为 BatchNorm1d
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, l = x.size()  # (batch, channels, length)

        # 1D FFT (返回复数张量)
        ffted = torch.fft.rfft(x, norm="ortho")  # 改为 rfft (1D)
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)  # (batch, c, l//2+1, 2)

        # 调整维度以匹配 Conv1d 输入
        ffted = ffted.permute(0, 1, 3, 2).contiguous()  # (batch, c, 2, l//2+1)
        ffted = ffted.view((batch, -1, ffted.size(-1)))  # (batch, c*2, l//2+1)

        # 1D 卷积处理
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        # 恢复复数形式
        ffted = ffted.view((batch, -1, 2, ffted.size(-1))).permute(0, 1, 3, 2).contiguous()
        ffted = torch.view_as_complex(ffted)

        # 1D 逆 FFT
        output = torch.fft.irfft(ffted, n=l, norm="ortho")  # 改为 irfft (1D)

        return output


class Freq_Fusionv1(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1, 3, 5, 7],  # 1D convolution kernel sizes
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4,
    ):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim * scale_ratio // spilt_num

        # Create parallel convolution branches with different kernel sizes
        self.conv_branches_1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, ks, padding=ks // 2),  # Add padding to maintain size
                nn.GELU(),
            ) for ks in kernel_size
        ])

        self.conv_branches_2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, ks, padding=ks // 2),
                nn.GELU(),
            ) for ks in kernel_size
        ])

        # Fusion layer for combining outputs from different kernel sizes
        self.conv_fuse_1 = nn.Sequential(
            nn.Conv1d(dim * len(kernel_size), dim, 1),
            nn.GELU(),
        )
        self.conv_fuse_2 = nn.Sequential(
            nn.Conv1d(dim * len(kernel_size), dim, 1),
            nn.GELU(),
        )

        self.conv_mid = nn.Sequential(
            nn.Conv1d(dim * 2, dim, 1),
            nn.GELU(),
        )
        self.FFC = FourierUnit(dim * 2, dim * 2)

        self.bn = nn.BatchNorm1d(dim * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)

        # Process each branch with different kernel sizes and concatenate
        x_1_branches = [branch(x_1) for branch in self.conv_branches_1]
        x_1 = self.conv_fuse_1(torch.cat(x_1_branches, dim=1))

        x_2_branches = [branch(x_2) for branch in self.conv_branches_2]
        x_2 = self.conv_fuse_2(torch.cat(x_2_branches, dim=1))

        x0 = torch.cat([x_1, x_2], dim=1)
        x = self.FFC(x0) + x0  # Residual connection
        x = self.relu(self.bn(x))
        return x

class Freq_Fusion(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1, 3, 5, 7],  # 1D 卷积核大小
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4,
    ):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim * scale_ratio // spilt_num

        # 改为 Conv1d
        self.conv_init_1 = nn.Sequential(
            nn.Conv1d(dim, dim, 1),  # 1x1 卷积
            nn.GELU(),
        )
        self.conv_init_2 = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.GELU(),
        )
        self.conv_mid = nn.Sequential(
            nn.Conv1d(dim * 2, dim, 1),
            nn.GELU(),
        )
        self.FFC = FourierUnit(dim * 2, dim * 2)  # 使用修改后的 1D FFT 模块

        self.bn = nn.BatchNorm1d(dim * 2)  # 改为 BatchNorm1d
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        x0 = torch.cat([x_1, x_2], dim=1)
        x = self.FFC(x0) + x0  # 残差连接
        x = self.relu(self.bn(x))
        return x

class Fused_Fourier_Conv_Mixer(nn.Module):
    def __init__(
        self,
        dim,
        token_mixer_for_global=Freq_Fusionv1,
        mixer_kernel_size=[1, 3, 5, 7],  # 1D 卷积核
        local_size=8,
    ):
        super(Fused_Fourier_Conv_Mixer, self).__init__()
        self.dim = dim
        self.mixer_global = token_mixer_for_global(
            dim=self.dim,
            kernel_size=mixer_kernel_size,
            se_ratio=8,
            local_size=local_size,
        )

        # 改为 Conv1d
        self.ca_conv = nn.Sequential(
            nn.Conv1d(2 * dim, dim, 1),
            nn.Conv1d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                padding_mode="reflect",  # 1D reflect padding
            ),
            nn.GELU(),
        )

        # 1D 通道注意力
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 1D 全局平均池化
            nn.Conv1d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # 改为 Conv1d
        self.conv_init = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1),
            nn.GELU(),
        )

        # 1D 深度可分离卷积
        self.dw_conv_1 = nn.Sequential(
            nn.Conv1d(
                self.dim,
                self.dim,
                kernel_size=3,
                padding=3 // 2,
                groups=self.dim,
                padding_mode="reflect",
            ),
            nn.GELU(),
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv1d(
                self.dim,
                self.dim,
                kernel_size=5,
                padding=5 // 2,
                groups=self.dim,
                padding_mode="reflect",
            ),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local_1 = self.dw_conv_1(x[0])  # 3x3
        x_local_2 = self.dw_conv_2(x[0])  # 5x5
        x_global = self.mixer_global(torch.cat([x_local_1, x_local_2], dim=1))
        x = self.ca_conv(x_global)
        x = self.ca(x) * x  # 通道注意力
        return x



class FourierAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 d_model=512):
        super().__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.d_model = d_model
        self.head_dim = d_model // 8  # 默认8个头

        # 核心傅里叶处理模块（直接处理1D时序）
        self.fourier_mixer = Fused_Fourier_Conv_Mixer(
            dim=self.head_dim,  # 使用每个头的维度
            token_mixer_for_global=Freq_Fusion
        )

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, D = queries.shape  # (batch_size, sequence_length, d_model)
        H = D // self.head_dim  # 计算头数

        # 重塑为多头形式 (B, L, H, E) -> (B, H, L, E)
        queries = queries.view(B, L, H, self.head_dim)
        queries = queries.permute(0, 2, 1, 3).contiguous()  # (B, H, L, E)
        queries = queries.view(B * H, L, self.head_dim)  # (B*H, L, E)

        # 调整形状以匹配 Conv1d 输入：(B*H, E, L)
        x_1d = queries.permute(0, 2, 1)  # (B*H, E, L)

        # 傅里叶混合处理（直接处理1D数据）
        x_out = self.fourier_mixer(x_1d)  # (B*H, E, L)

        # 恢复原始形状 (B*H, L, E)
        x_out = x_out.permute(0, 2, 1).contiguous()  # (B*H, L, E)
        x_out = x_out.view(B, H, L, self.head_dim)  # (B, H, L, E)
        x_out = x_out.permute(0, 2, 1, 3).contiguous()  # (B, L, H, E)
        x_out = x_out.view(B, L, D)  # (B, L, D)

        if self.output_attention:
            # 输出一个均匀注意力（因为傅里叶混合是全局操作）
            attn = torch.ones(B, H, L, L, device=x_out.device) * (1.0 / L)
            return x_out, attn
        return x_out, None

class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=False), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out


class TimeAwareFiLM(nn.Module):
    def __init__(self, static_dim, dynamic_dim, hidden_dim, time_dim=16):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.Mish(),  # 添加激活函数
            nn.Linear(hidden_dim, dynamic_dim * 2)  # 输出scale和shift的总维度
        )

    def forward(self, static, dynamic):
        # static: [B, S], dynamic: [B, T, D]
        static_expanded = static.unsqueeze(1).expand(-1, dynamic.size(1), -1)  # [B, T, S]

        params = self.static_encoder(static_expanded)
        scale, shift = torch.split(params, dynamic.size(-1), dim=-1)  # 各为[B, 10]

        return dynamic * scale + shift

class FiLM(nn.Module):
    """
    Refers to FiLM: Visual Reasoning with a General Conditioning Layer(AAAI 2018)
    <https://ojs.aaai.org/index.php/AAAI/article/view/11671>

    """

    def __init__(self, input_dim, condition_dim):
        super(FiLM, self).__init__()


        self.fc_gamma = nn.Linear(condition_dim, input_dim)
        self.fc_beta = nn.Linear(condition_dim, input_dim)

    def forward(self, x, condition):

        gamma = self.fc_gamma(condition)
        beta = self.fc_beta(condition)

        y = gamma * x + beta
        return y

