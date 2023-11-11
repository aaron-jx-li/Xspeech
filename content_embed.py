import torch
import numpy as np

EMBED_PATH = "./data/vctk/hubert-base-1w/"
embed = torch.load(EMBED_PATH + "ninth_layer_hidden/p225_003.hid")
index = torch.load(EMBED_PATH + "cluster_index/p225_003.ind")
print(embed.shape)
print(index.shape)