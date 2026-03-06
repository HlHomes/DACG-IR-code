import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Adaptive_Gated_Fusion(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super(Adaptive_Gated_Fusion, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, in_dim), num_channels=in_dim) if in_dim > 1 else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, in_dim),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, self.out_dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, f_enc, f_dec):
        combined = torch.cat([f_enc, f_dec], dim=1)
        spatial_logit = self.spatial_gate(combined)

        b, c, _, _ = combined.shape
        y = self.avg_pool(combined).view(b, c)
        channel_logit = self.channel_gate(y).view(b, self.in_dim, 1, 1)

        atten_weight = torch.sigmoid(spatial_logit + channel_logit)

        f_enc_filtered = f_enc * atten_weight

        out = torch.cat([f_enc_filtered, f_dec], dim=1)
        out = self.fusion_conv(out)

        return out


class Degradation_Aware_Module(nn.Module):

    def __init__(self, dim, num_scales=3, dim_list=[48, 96, 192, 384]):
        super().__init__()

        inter_dim = dim
        context_dim = dim * 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, inter_dim, kernel_size=3, padding=1, stride=1),
            nn.GELU()
        )
        self.scale_branches = nn.ModuleList()
        for s in range(num_scales):
            # kernel sizes: 3x3, 5x5, 7x7...
            k_size = 2 * s + 3
            branch = nn.Sequential(
                nn.Conv2d(inter_dim, inter_dim, kernel_size=k_size, padding=k_size // 2, groups=inter_dim),
                nn.Conv2d(inter_dim, inter_dim, kernel_size=1)
            )
            self.scale_branches.append(branch)

        fusion_in_dim = inter_dim * num_scales

        self.fusion = nn.Conv2d(fusion_in_dim, context_dim, kernel_size=1)

        self.spatial_gate = nn.Conv2d(context_dim, context_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.global_process = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim * 2),
            nn.LayerNorm(context_dim * 2),
            nn.GELU(),
            nn.Linear(context_dim * 2, context_dim)
        )

        self.prompt_layers = nn.ModuleList()
        curr_in = context_dim
        for out_dim in dim_list:
            self.prompt_layers.append(
                nn.Sequential(
                    nn.Linear(curr_in, curr_in),
                    nn.LayerNorm(curr_in),
                    nn.GELU(),
                    nn.Linear(curr_in, out_dim)
                )
            )
            curr_in = out_dim

    def forward(self, x):
        x_feat = self.stem(x)  # [B, C, H, W]

        scale_feats = [branch(x_feat) for branch in self.scale_branches]
        scale_feats = torch.cat(scale_feats, dim=1)  # [B, C*num_scales, H, W]

        feat = self.fusion(scale_feats)  # [B, 2C, H, W]

        gate_map = self.sigmoid(self.spatial_gate(feat))
        feat = feat * gate_map  # [B, 2C, H, W]

        feat_avg = torch.mean(feat, dim=(2, 3))  # [1,2C]
        feat_std = torch.std(feat, dim=(2, 3))  # [1,2C]
        global_stat = torch.cat([feat_avg, feat_std], dim=1)  # [1,4C]

        global_feat = self.global_process(global_stat)

        layer_prompts = []
        z = global_feat
        for layer in self.prompt_layers:
            z = layer(z)
            layer_prompts.append(z)

        return layer_prompts, global_feat


class Context_Gating_DualDomain_Modulation(nn.Module):

    def __init__(self, dim, context_dim=None):
        super().__init__()
        self.dim = dim
        self.context_dim = 2 * context_dim
        self.freq_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1)
        )
        self.context_mapper = nn.Sequential(
            nn.Linear(self.context_dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim * 2)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x, global_feat):
        B, C, H, W = x.shape

        spatial_feat = self.spatial_conv(x)

        x_32 = x.to(torch.float32)
        x_fft = torch.fft.rfft2(x_32, norm='ortho')

        real = x_fft.real
        imag = x_fft.imag
        f_cat = torch.cat([real, imag], dim=1)

        f_feat = self.freq_conv(f_cat)

        scale = self.context_mapper(global_feat)
        scale = scale.to(torch.float32)
        scale = torch.sigmoid(scale).unsqueeze(-1).unsqueeze(-1)  # [B, 2*C, 1, 1]

        f_weighted = f_feat * scale

        w_real, w_imag = torch.chunk(f_weighted, 2, dim=1)
        w_complex = torch.complex(w_real, w_imag)
        freq_spatial = torch.fft.irfft2(w_complex, s=(H, W), norm='ortho')
        freq_spatial = freq_spatial.to(x.dtype)

        out = torch.cat([spatial_feat, freq_spatial], dim=1)
        out = self.fusion(out)

        return out + x


class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Context_Adaptive_Gated_Attention(nn.Module):

    def __init__(self, dim, num_heads, bias, context_dim):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.log_base_temperature = nn.Parameter(torch.zeros(num_heads, 1, 1))

        self.temp_adapter = nn.Sequential(
            nn.Linear(context_dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, num_heads)
        )

        self.attn_output_gate = nn.Sequential(
            nn.Linear(context_dim, dim),
        )

    def forward(self, x, context_emb):
        b, c, h, w = x.shape

        log_delta = self.temp_adapter(context_emb).view(b, self.num_heads, 1, 1)
        log_temp = self.log_base_temperature + log_delta
        total_temp = torch.exp(log_temp)  # [B, heads, 1, 1]

        gate_score = self.attn_output_gate(context_emb)  # [B, C]
        gate_score = gate_score.view(b, self.num_heads, self.head_dim, 1)  # [B, heads, C_head, 1]
        gate_score = torch.sigmoid(gate_score)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn * total_temp
        attn = attn.softmax(dim=-1)

        out = (attn @ v)  # [B, heads, C_head, HW]

        out = out * gate_score

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=h, w=w)
        out = self.project_out(out)

        return out


class Context_Gate_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, context_dim):
        super(Context_Gate_TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Context_Adaptive_Gated_Attention(dim, num_heads, bias, context_dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x, context_emb):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, context_emb)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


class DACG_IR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 num_scales=3,
                 ):
        super().__init__()

        self.dim = dim
        self.num_scales = num_scales
        self.dim_list = [int(dim * 2 ** i) for i in range(4)]
        self.padder_size = 2 ** len(num_blocks)

        self.context_net = Degradation_Aware_Module(
            dim=self.dim,
            num_scales=self.num_scales,
            dim_list=self.dim_list
        )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        # Encoder
        self.encoder_level1 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[0], heads[0], ffn_expansion_factor, bias, LayerNorm_type,
                                          self.dim_list[0])
            for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(self.dim_list[0])
        self.encoder_level2 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[1], heads[1], ffn_expansion_factor, bias, LayerNorm_type,
                                          self.dim_list[1])
            for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(self.dim_list[1])
        self.encoder_level3 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[2], heads[2], ffn_expansion_factor, bias, LayerNorm_type,
                                          self.dim_list[2])
            for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(self.dim_list[2])
        self.latent = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[3], heads[3], ffn_expansion_factor, bias, LayerNorm_type,
                                          self.dim_list[3])
            for _ in range(num_blocks[3])
        ])

        self.freq_fusion = Context_Gating_DualDomain_Modulation(dim=self.dim_list[3], context_dim=self.dim)

        self.up4_3 = Upsample(self.dim_list[3])
        self.skip_fusion3 = Adaptive_Gated_Fusion(in_dim=self.dim_list[2])

        # Decoder
        self.decoder_level3 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[2], heads[2], ffn_expansion_factor, bias, LayerNorm_type,
                                          self.dim_list[2])
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(self.dim_list[2])
        self.skip_fusion2 = Adaptive_Gated_Fusion(in_dim=self.dim_list[1])

        self.decoder_level2 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[1], heads[1], ffn_expansion_factor, bias, LayerNorm_type,
                                          self.dim_list[1])
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(self.dim_list[1])
        self.skip_fusion1 = Adaptive_Gated_Fusion(in_dim=self.dim_list[0], out_dim=self.dim_list[1])  # 48 -> 96

        self.decoder_level1 = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[1], heads[0], ffn_expansion_factor, bias, LayerNorm_type,
                                          self.dim_list[1])
            for _ in range(num_blocks[0])
        ])

        # Refinement
        self.refinement = nn.ModuleList([
            Context_Gate_TransformerBlock(self.dim_list[1], heads[0], ffn_expansion_factor, bias, LayerNorm_type,
                                          self.dim_list[1])
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(self.dim_list[1], out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        B, C, H, W = inp_img.shape
        inp_img = self.check_image_size(inp_img)

        layer_prompts, global_feat = self.context_net(inp_img)
        p1, p2, p3, p4 = layer_prompts
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = inp_enc_level1
        for block in self.encoder_level1:
            out_enc_level1 = block(out_enc_level1, p1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level2:
            out_enc_level2 = block(out_enc_level2, p2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = inp_enc_level3
        for block in self.encoder_level3:
            out_enc_level3 = block(out_enc_level3, p3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = inp_enc_level4
        for block in self.latent:
            latent = block(latent, p4)
        latent = self.freq_fusion(latent, global_feat)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = self.skip_fusion3(out_enc_level3, inp_dec_level3)
        out_dec_level3 = inp_dec_level3
        for block in self.decoder_level3:
            out_dec_level3 = block(out_dec_level3, p3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.skip_fusion2(out_enc_level2, inp_dec_level2)
        out_dec_level2 = inp_dec_level2
        for block in self.decoder_level2:
            out_dec_level2 = block(out_dec_level2, p2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.skip_fusion1(out_enc_level1, inp_dec_level1)
        out_dec_level1 = inp_dec_level1
        for block in self.decoder_level1:
            out_dec_level1 = block(out_dec_level1, p2)
        for block in self.refinement:
            out_dec_level1 = block(out_dec_level1, p2)

        out = self.output(out_dec_level1) + inp_img

        return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # 需要填充的高度和宽度，以确保图像的高度和宽度是 self.padder_size 的倍数。
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x


##########################################################################
# 测试代码
##########################################################################
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DACG_IR(
        dim=32,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        num_scales=3,
    ).to(device)

    # 测试输入
    inp = torch.randn(1, 3, 256, 256).to(device)

    # 运行推理
    out = model(inp)
    print(f"Input shape: {inp.shape}")
    print(f"Output shape: {out.shape}")

    # 检查参数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {model_params / 1e6:.3f} M")

    flops = FlopCountAnalysis(model, inp)
    print("\nFLOPs Analysis:")
    print(flop_count_table(flops))

    # 简单显存测试
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.2f} MB")