import os
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        do_checkpoint=True,
        relative_pos_embeddings=False,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
    

class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False,
                 mean=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            return h[:, :, 0]
        

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 64)
        # outpur dimension = # global actions
        self.fc4 = nn.Linear(64, num_labels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        out = F.softmax(x, dim=0)
        return out
    
