import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN

_HAS_FLASH_ATTN = False
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward

        _HAS_FLASH_ATTN = True
    except Exception:
        pass
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    vq_z: Optional[torch.Tensor] = None
    vq_indices: Optional[torch.Tensor] = None
    vq_loss: Optional[torch.Tensor] = None
    mel_recon: Optional[torch.Tensor] = None
    chroma_recon: Optional[torch.Tensor] = None
    ctc_logits: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.LongTensor] = None


@dataclass
class Config:
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    gradient_checkpointing: bool = False
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    attention_bias: bool = False
    sliding_window: Optional[int] = None
    mono: bool = True
    sample_rate: int = 24000
    n_fft: int = 2048
    hop_length: int = 240
    win_length: int = 2048
    add_mel: bool = True
    n_mels: int = 128
    add_chroma: bool = True
    n_chroma: int = 12
    add_ctc: bool = True
    text_vocab_size: int = 250003
    ctc_blank_id: int = 250002
    add_vq: bool = True
    vq_codebook_dim: int = 1024
    vq_codebook_size: int = 32768
    vq_layer_idx: int = 15
    conv_depthwise_kernel_size: int = 31
    conv_dropout: float = 0.1
    activation_dropout: float = 0.1
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    self_attn_dropout: float = 0.1


class FeatureProcessor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        window = torch.hann_window(config.n_fft)
        self.register_buffer("window", window)

        mel_fb = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=0.0,
            fmax=None,
            htk=False,
            norm="slaney",
            dtype=np.float32,
        )
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb))
        chroma_fb = librosa.filters.chroma(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_chroma=config.n_chroma,
            tuning=0.0,
            ctroct=5.0,
            octwidth=2.0,
            norm=2,
            base_c=True,
            dtype=np.float32,
        )
        self.register_buffer("chroma_fb", torch.from_numpy(chroma_fb))

        self.multiplier = 10.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, waveform: torch.Tensor, top_db: Optional[float] = 80.0):
        shape = waveform.size()
        waveform = waveform.reshape(-1, shape[-1])
        spec_f = self.spectrogram(waveform)
        spec_f = spec_f.reshape(
            shape[:-1] + spec_f.shape[-2:]
        )  # [batch, (channel)] + [freq, time]
        mel = torch.einsum("cf,...ft->...ct", self.mel_fb, spec_f)
        chroma = torch.einsum("cf,...ft->...ct", self.chroma_fb, spec_f)
        mel = self.power_to_db(mel, top_db=top_db)
        chroma = self.power_to_db(chroma, top_db=top_db)
        return mel, chroma

    def spectrogram(self, waveform: torch.Tensor):
        spec_f = (
            torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            .abs()
            .pow(2.0)
        )
        return spec_f

    def power_to_db(self, x: torch.Tensor, top_db: Optional[float] = 80.0):
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        if top_db is not None:
            shape = x_db.size()
            packed_channels = shape[-3] if x_db.dim() > 2 else 1
            x_db = x_db.reshape(-1, packed_channels, shape[-2], shape[-1])

            x_db = torch.max(
                x_db, (x_db.amax(dim=(-3, -2, -1)) - top_db).view(-1, 1, 1, 1)
            )

            x_db = x_db.reshape(shape)

        return x_db


def drop_path(
    input: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=input.dtype, device=input.device
    )
    random_tensor.floor_()
    output = input.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class GRN1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        global_features = torch.norm(hidden_states, p=2, dim=1, keepdim=True)
        norm_features = global_features / (
            global_features.mean(dim=-1, keepdim=True) + 1e-6
        )
        hidden_states = (
            self.weight * (hidden_states * norm_features) + self.bias + hidden_states
        )

        return hidden_states


class ConvNeXt1dLayer(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, hidden_act: str = "gelu"):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = ACT2FN[hidden_act]
        self.grn = GRN1d(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, hidden_states: torch.FloatTensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            hidden_states = hidden_states.masked_fill(~mask.bool()[:, None, :], 0.0)
        residual = hidden_states
        x = self.dwconv(hidden_states)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)
        x = residual + self.drop_path(x)
        return x


class Conv1dSubsampling(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        base_channels: int = 256,
        out_channels: int = 1024,
    ):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, base_channels, 3, 1, 1)
        self.conv_2 = nn.Conv1d(base_channels, base_channels * 2, 3, 2, 1)
        self.conv_3 = nn.Conv1d(base_channels * 2, base_channels * 3, 3, 2, 1)
        self.convnext_1 = nn.ModuleList(
            [
                ConvNeXt1dLayer(dim=base_channels, drop_path=0.0, hidden_act="gelu")
                for _ in range(3)
            ]
        )
        self.convnext_2 = nn.ModuleList(
            [
                ConvNeXt1dLayer(dim=base_channels * 2, drop_path=0.0, hidden_act="gelu")
                for _ in range(3)
            ]
        )
        self.convnext_3 = nn.ModuleList(
            [
                ConvNeXt1dLayer(dim=base_channels * 3, drop_path=0.0, hidden_act="gelu")
                for _ in range(3)
            ]
        )
        self.layer_norm_1 = nn.LayerNorm(base_channels, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(base_channels, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(base_channels * 2, eps=1e-6)
        self.layer_norm_4 = nn.LayerNorm(base_channels * 3, eps=1e-6)
        self.linear = nn.Linear(base_channels * 3, out_channels)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = rearrange(x, "b 1 c t -> b c t")
        if mask is not None:
            x = x.masked_fill(~mask.bool()[:, None, :], 0.0)
        x = self.conv_1(x)
        x = self.layer_norm_1(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.convnext_1:
            x = layer(x, mask)
        x = self.layer_norm_2(x.transpose(1, 2)).transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(~mask.bool()[:, None, :], 0.0)
            mask = mask[:, ::2] if mask is not None else None
        x = self.conv_2(x)
        for layer in self.convnext_2:
            x = layer(x, mask)
        x = self.layer_norm_3(x.transpose(1, 2)).transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(~mask.bool()[:, None, :], 0.0)
            mask = mask[:, ::2] if mask is not None else None
        x = self.conv_3(x)
        for layer in self.convnext_3:
            x = layer(x, mask)
        x = self.layer_norm_4(x.transpose(1, 2))
        x = self.linear(x)
        return x, mask


class Conv1dUpsampling(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        base_channels: int = 256,
        out_channels: int = 128,
    ):
        super().__init__()
        self.conv_1 = nn.Conv1d(base_channels * 2, base_channels * 2, 3, 1, 1)
        self.conv_2 = nn.Conv1d(base_channels * 2, base_channels, 3, 1, 1)
        self.conv_3 = nn.Conv1d(base_channels, out_channels, 3, 1, 1)
        self.convnext_1 = nn.ModuleList(
            [
                ConvNeXt1dLayer(dim=base_channels * 2, drop_path=0.0, hidden_act="gelu")
                for _ in range(3)
            ]
        )
        self.convnext_2 = nn.ModuleList(
            [
                ConvNeXt1dLayer(dim=base_channels, drop_path=0.0, hidden_act="gelu")
                for _ in range(3)
            ]
        )
        self.linear = nn.Linear(in_channels, base_channels * 2)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.linear(x).transpose(1, 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if mask is not None:
            mask = mask.repeat_interleave(2, dim=1)
            x = x.masked_fill(~mask.bool()[:, None, :], 0.0)
        x = self.conv_1(x)
        for layer in self.convnext_1:
            x = layer(x, mask)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if mask is not None:
            mask = mask.repeat_interleave(2, dim=1)
            x = x.masked_fill(~mask.bool()[:, None, :], 0.0)
        x = self.conv_2(x)
        for layer in self.convnext_2:
            x = layer(x, mask)
        if mask is not None:
            x = x.masked_fill(~mask.bool()[:, None, :], 0.0)
        x = self.conv_3(x)
        x = rearrange(x, "b c t -> b 1 c t")
        return x, mask


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q, k, cos, sin, unsqueeze_dim=1
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_type = "default"
        dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.config = config

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )

        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=2
            )

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_dtype("cuda")
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        if _HAS_FLASH_ATTN:
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=dropout_rate,
                sliding_window=self.config.sliding_window,
                use_top_left_mask=False,
                is_causal=False,
            )
        else:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout_rate,
                is_causal=False,
            )
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output


class Codebook(nn.Module):
    def __init__(self, codebook_size, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps

        weight = torch.randn(codebook_size, codebook_dim, dtype=torch.float32)
        weight.uniform_(-1.0 / math.sqrt(codebook_size), 1.0 / math.sqrt(codebook_size))

        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(
            torch.zeros(codebook_size, dtype=torch.float32) + 8, requires_grad=False
        )
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size,
        codebook_dim,
        distance="l2",
        decay=0.99,
        distributed=True,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.distance = distance
        if self.distance == "l2":
            self.distance_fn = self.distance_l2
        elif self.distance == "cosine":
            self.distance_fn = self.distance_cosine
        else:
            raise ValueError(f"Invalid distance: {self.distance}")
        self.decay = decay
        self.distributed = distributed
        self.embedding = Codebook(self.codebook_size, self.codebook_dim, decay=decay)

    def distance_l2(self, z: torch.Tensor):
        d = (
            -torch.sum(z.detach() ** 2, dim=1, keepdim=True)
            - torch.sum(self.embedding.weight**2, dim=1)
            + 2
            * torch.einsum(
                "bd, dn-> bn",
                z.detach(),
                rearrange(self.embedding.weight, "n d-> d n"),
            )
        )
        return d

    def distance_cosine(self, z: torch.Tensor):
        normed_z_flattened = F.normalize(z, dim=1, p=2.0).detach()
        normed_codebook = F.normalize(self.embedding.weight, dim=1, p=2.0)
        d = torch.einsum(
            "bd,dn->bn",
            normed_z_flattened,
            rearrange(normed_codebook, "n d -> d n"),
        )
        return d

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, z):
        z = z.to(torch.float32)
        b, t, h = z.size()
        z_flattened = rearrange(z, "b t h -> (b t) h").contiguous()

        d = self.distance_fn(z_flattened)

        encoding_indices = torch.argmax(d, dim=1)
        z_q = self.embedding(encoding_indices)
        z_q = rearrange(z_q, "(b t) h -> b t h", b=b)

        loss = torch.mean((z_q.detach() - z) ** 2)
        encoding_indices = rearrange(encoding_indices, "(b t) -> b t", b=b)

        return z_q, encoding_indices, loss


class ConvolutionModule(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError(
                "`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding"
            )

        self.layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, bias=True
        )
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=(config.conv_depthwise_kernel_size - 1) // 2,
            groups=config.hidden_size,
        )

        self.depthwise_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            bias=True,
        )
        self.activation = ACT2FN[config.hidden_act]
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.dropout = nn.Dropout(config.conv_dropout)

        self.config = config

    def forward(self, hidden_states, conv_mask=None):
        hidden_states = self.layer_norm(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.glu(hidden_states)

        if conv_mask is not None:
            hidden_states = hidden_states.masked_fill(
                ~conv_mask.bool().unsqueeze(1), 0.0
            )

        hidden_states = self.depthwise_conv(hidden_states)

        hidden_states = self.depthwise_layer_norm(
            hidden_states.transpose(1, 2)
        ).transpose(1, 2)

        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

        self.config = config

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class ConformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ffn1_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            bias=True,
        )
        self.ffn1 = FeedForward(config)

        self.self_attn_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            bias=True,
        )
        self.self_attn_dropout = nn.Dropout(config.self_attn_dropout)
        self.self_attn = Attention(config=config)

        self.conv_module = ConvolutionModule(config)

        self.ffn2_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            bias=True,
        )
        self.ffn2 = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            bias=True,
        )

        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        conv_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, conv_mask=conv_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual

        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [ConformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.rotary_emb = RotaryEmbedding(config=config)
        if config.add_vq:
            self.vq = VectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
            )
            self.vq_in = nn.Conv1d(
                config.hidden_size,
                config.vq_codebook_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.vq_out = nn.Conv1d(
                config.vq_codebook_dim,
                config.hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.vq_bn = nn.BatchNorm1d(
                config.vq_codebook_dim, affine=False, momentum=0.05
            )
        self.config = config


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.feature_processor = FeatureProcessor(config)
        self.feature_extractor = Conv1dSubsampling(
            in_channels=config.n_mels,
            base_channels=256,
            out_channels=config.hidden_size,
        )
        self.mel_bn = nn.BatchNorm2d(config.n_mels, affine=False, momentum=0.1)
        self.chroma_bn = nn.BatchNorm2d(config.n_chroma, affine=False, momentum=0.1)
        self.encoder = Encoder(config)

        if config.add_ctc:
            self.ctc_head = nn.Linear(
                config.hidden_size, config.text_vocab_size, bias=True
            )
        if config.add_mel:
            self.mel_head = Conv1dUpsampling(
                in_channels=config.hidden_size,
                base_channels=256,
                out_channels=config.n_mels,
            )
        if config.add_chroma:
            self.chroma_head = Conv1dUpsampling(
                in_channels=config.hidden_size,
                base_channels=256,
                out_channels=config.n_chroma,
            )

        self.config = config
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(
                    module.groups / (module.in_channels * module.kernel_size[0])
                )
                nn.init.uniform_(module.bias, a=-k, b=k)

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def wav2features(self, wav: torch.Tensor):
        mel, chroma = self.feature_processor(wav)
        mel = mel[..., :-1]
        chroma = chroma[..., :-1]
        mel = self.mel_bn(rearrange(mel, "b c f t -> b f c t"))
        chroma = self.chroma_bn(rearrange(chroma, "b c f t -> b f c t"))
        mel = rearrange(mel, "b f c t -> b c f t")
        chroma = rearrange(chroma, "b f c t -> b c f t")
        return mel, chroma

    @torch.no_grad()
    def quantize(
        self,
        wav: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        mask: Optional[torch.LongTensor] = None,
    ):
        if wav is not None:
            mel, _ = self.wav2features(wav)
        hidden_states, attention_mask = self.feature_extractor(mel, mask=mask)
        hidden_states = self.encoder.dropout(hidden_states)

        position_ids = torch.arange(
            0, hidden_states.shape[1], device=hidden_states.device
        ).unsqueeze(0)
        position_embeddings = self.encoder.rotary_emb(hidden_states, position_ids)

        for i, layer in enumerate(self.encoder.layers):
            if i == self.config.vq_layer_idx:
                hidden_states = rearrange(hidden_states, "b t c -> b c t")
                hidden_states = self.encoder.vq_in(hidden_states)
                hidden_states = self.encoder.vq_bn(hidden_states)
                hidden_states = rearrange(hidden_states, "b c t -> b t c")
                vq_z, vq_indices, vq_loss = self.encoder.vq(hidden_states)
                break

            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                conv_mask=attention_mask,
            )

        return vq_z, vq_indices, attention_mask

    def get_mel_recon(self, x: torch.Tensor, conv_mask: Optional[torch.Tensor] = None):
        if self.config.add_mel:
            mel_recon, _ = self.mel_head(x, conv_mask)
            return mel_recon
        else:
            return None

    def get_chroma_recon(
        self, x: torch.Tensor, conv_mask: Optional[torch.Tensor] = None
    ):
        if self.config.add_chroma:
            chroma_recon, _ = self.chroma_head(x, conv_mask)
            return chroma_recon
        else:
            return None

    def get_ctc_logits(self, x: torch.Tensor):
        if self.config.add_ctc:
            return self.ctc_head(x)
        else:
            return None

    @torch.no_grad()
    def decode(
        self,
        z_q: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        return_ctc_logits: bool = False,
        return_mel_recon: bool = False,
        return_chroma_recon: bool = False,
    ) -> EncoderOutput:
        if indices is not None:
            z_q = self.encoder.vq.embedding(indices)

        hidden_states = self.encoder.vq_out(rearrange(z_q, "b t d -> b d t"))
        hidden_states = rearrange(hidden_states, "b d t -> b t d")

        position_ids = torch.arange(
            0, hidden_states.shape[1], device=hidden_states.device
        ).unsqueeze(0)
        position_embeddings = self.encoder.rotary_emb(hidden_states, position_ids)

        for i, layer in enumerate(self.encoder.layers):
            if i < self.config.vq_layer_idx:
                continue
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                conv_mask=attention_mask,
            )
        output = EncoderOutput(last_hidden_state=hidden_states)
        if return_ctc_logits:
            output.ctc_logits = self.get_ctc_logits(hidden_states)
        if return_mel_recon:
            output.mel_recon = self.get_mel_recon(hidden_states, attention_mask)
        if return_chroma_recon:
            output.chroma_recon = self.get_chroma_recon(hidden_states, attention_mask)
        return output
