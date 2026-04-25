import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer, MultiScaleEncoderLayer, FourierEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, FiLM, TimeAwareFiLM, DiffAttention
from layers.Embed import DataEmbedding_inverted
from einops import reduce
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.static_numbers = configs.static_numbers
        self.dynamic_numbers = configs.enc_in - self.static_numbers
        # self.film = FiLM(self.static_numbers, self.dynamic_numbers, hidden_dim=64)
        self.film = TimeAwareFiLM(self.static_numbers, self.dynamic_numbers, hidden_dim=64)

        self.dynamic_embed = nn.Linear(self.dynamic_numbers, configs.d_model)


        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.head_type = 'V0'  # V5
            if self.head_type == 'V0':
                self.act = F.gelu
                self.dropout = nn.Dropout(configs.dropout)
                self.projection = nn.Linear(configs.d_model * self.dynamic_numbers, configs.num_class)  # self.dynamic_numbers/configs.enc_in
            elif self.head_type == 'V1':
                self.cls_token = nn.Parameter(torch.randn(1, 1, configs.d_model))
                self.act = F.gelu
                self.dropout = nn.Dropout(configs.dropout)
                self.projection = nn.Linear(configs.d_model, configs.num_class)
            elif self.head_type in ['V2', 'V3']:
                self.act = F.gelu
                self.dropout = nn.Dropout(configs.dropout)
                self.projection = nn.Linear(configs.d_model, configs.num_class)
            elif self.head_type == 'V4':
                self.act = F.gelu
                self.dropout = nn.Dropout(configs.dropout)
                self.projection = nn.Linear(configs.d_model * 2, configs.num_class)
            elif self.head_type == 'V5':
                self.cls_token = nn.Parameter(torch.randn(1, configs.num_class, configs.d_model))
                self.act = F.gelu
                self.dropout = nn.Dropout(configs.dropout)
                self.projection = nn.Linear(configs.d_model * configs.num_class, configs.num_class)
            elif self.head_type == 'V6':
                self.cls_token = nn.Parameter(torch.randn(1, configs.num_class, configs.d_model))
                self.act = F.gelu
                self.dropout = nn.Dropout(configs.dropout)
                self.projection = nn.Linear(configs.d_model, 1)
            elif self.head_type == 'V7':
                self.cls_token = nn.Parameter(torch.randn(1, configs.num_class, configs.d_model))
                self.act = F.gelu
                self.dropout = nn.Dropout(configs.dropout)
                # self.projection = nn.Linear(configs.d_model, 1)
                self.w = nn.Parameter(torch.randn(configs.d_model, configs.num_class))
                self.b = nn.Parameter(torch.randn(1, configs.num_class))

    # 从原始输入中分离静态变量
    def split_static_dynamic(self, x):
        static = x[:, 0, -self.static_numbers:]  # 取第一个时间步的静态变量 [B, 5]
        dynamic = x[:, :, 0:self.dynamic_numbers]  # 所有时间步的动态变量 [B, T, 15]
        return static, dynamic

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()    # TODO
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):

        static, dynamic = self.split_static_dynamic(x_enc)
        # # Embedding
        modulated_dynamic = self.film(static, dynamic) + dynamic

        enc_out = self.enc_embedding(modulated_dynamic, None)  # modulated_dynamic
        if self.head_type in ['V1', 'V5', 'V6', 'V7']:
            cls_token = self.cls_token.repeat(enc_out.shape[0], 1, 1)  # (B, N+1, D)
            enc_out = torch.cat([enc_out, cls_token], dim=1)

        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # enc_out = self.film(static, enc_out)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)  # TODO
        if self.head_type == 'V0':
            output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
            output = self.projection(output)
        elif self.head_type == 'V1':
            output = output[:, -1]
            output = self.projection(output)
        elif self.head_type == 'V2':
            output = reduce(output, 'b n d -> b d', reduction='mean')
            output = self.projection(output)
        elif self.head_type == 'V3':
            output = reduce(output, 'b n d -> b d', reduction='max')
            output = self.projection(output)
        elif self.head_type == 'V4':
            output1 = reduce(output, 'b n d -> b d', reduction='mean')
            output2 = reduce(output, 'b n d -> b d', reduction='max')
            output = torch.cat([output1, output2], dim=1)
            output = self.projection(output)
        elif self.head_type == 'V5':
            output = output[:, -cls_token.shape[1]:]   # [B, C, D]
            output = output.reshape(output.shape[0], -1)  # [B, (C D)]
            output = self.projection(output)
        elif self.head_type == 'V6':
            output = output[:, -cls_token.shape[1]:]
            output = output.reshape(-1, output.shape[-1])  # [(B C), D)]
            output = self.projection(output)
            output = output.reshape(-1, cls_token.shape[1])
        elif self.head_type == 'V7':
            output = output[:, -cls_token.shape[1]:]
            output = torch.einsum('bcd,dc->bc', output, self.w)
            output = output + self.b

        # output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
