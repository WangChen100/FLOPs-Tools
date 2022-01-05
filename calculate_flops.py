
import torch

import _init_paths
from extra.flop_count import flop_count
from models.modules.basic_block import BasicBlock 
from models.modules.transformer_block import GeneralTransformerBlock

planes = 32
input_size=[1, planes, 96, 72]

dump_input = torch.rand(input_size)

# basicblock = BasicBlock(planes,planes)
transblock = GeneralTransformerBlock(planes,planes,num_heads=8,window_size=[12,9])

# print("FLOPs of basicblock is: ")
# print(flop_count(basicblock, (dump_input,)))
B, C, H, W = dump_input.size()
x = dump_input.view(B, C, -1).permute(0, 2, 1).contiguous()  # reshape
print("FLOPs of atten block is: ")
print(flop_count(transblock.attn, (x,torch.tensor(H),torch.tensor(W))))
print("FLOPs of mlp block is: ")
print(flop_count(transblock.mlp, (x,torch.tensor(H),torch.tensor(W))))
x_pad = transblock.attn.pad_helper.pad_if_needed(x.view(B,H,W,C), (B,H,W,C))
x_permute = transblock.attn.permute_helper.permute(x_pad, x_pad.size())
print("FLOPs of single attention is: ")
print(flop_count(transblock.attn.attn, (x_permute,x_permute,x_permute)))
print("FLOPs of transblock is: ")
print(flop_count(transblock, (dump_input,)))
