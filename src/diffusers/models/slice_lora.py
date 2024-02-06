# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch.nn.functional as F
from torch import nn
import math


class SliceLoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"SliceLoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        self.rank = rank
        self.dh = in_features
        self.a1 = int(self.dh/rank) # 256
        self.b1 = rank # 4
        self.a2 = rank # 4
        self.b2 = int(out_features/rank) 
        self.down = nn.Linear(self.a1, self.a2, bias=False, device=device, dtype=dtype) # A
        self.network_alpha = network_alpha
        self.up = nn.Linear(self.b1, self.b2, bias=False, device=device, dtype=dtype) # B

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        # print("SliceLoRA")
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        # print("hidden state", hidden_states.shape)
        
        if len(hidden_states.shape) == 3:
            B1, C, D = hidden_states.size() # get the matrix shape
            hidden_states = hidden_states.view(-1, self.a2,  self.a1)
            # print("hidden state2",hidden_states.shape)

            B2, _, _ = hidden_states.size() # get the matrix shape
            up_hidden_states = self.up(self.down(hidden_states.to(dtype)))
            # print("uphidden state1",up_hidden_states.shape)

            up_hidden_states= up_hidden_states.view(B1, C, self.b2*self.b1)
            # print("uphidden state2",up_hidden_states.shape)
            # exit()
        else: 
            B1, C = hidden_states.size() # get the matrix shape
            hidden_states = hidden_states.view(B1, self.a2, self.a1)
            hidden_states = hidden_states.view(B1*self.a2, self.a1)
            up_hidden_states = self.up((self.down(hidden_states.to(dtype))))
            up_hidden_states= up_hidden_states.view(B1, self.b1* self.b2)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank
        # print("huehue")
        # exit()
        return up_hidden_states.to(orig_dtype)


class SliceLoRAConv2dLayer(nn.Module):
    def __init__(
        self, in_features, out_features, rank=4, kernel_size=(1, 1), stride=(1, 1), padding=0, network_alpha=None
    ):
        super().__init__()

        self.down = nn.Conv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # according to the official kohya_ss trainer kernel_size are always fixed for the up layer
        # # see: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        self.up = nn.Conv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class SliceLoRACompatibleConv(nn.Conv2d):
    """
    A convolutional layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[SliceLoRAConv2dLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[SliceLoRAConv2dLayer]):
        self.lora_layer = lora_layer

    def forward(self, x):
        if self.lora_layer is None:
            # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
            # see: https://github.com/huggingface/diffusers/pull/4315
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return super().forward(x) + self.lora_layer(x)


class SliceLoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[SliceLoRALinearLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[SliceLoRAConv2dLayer]):
        self.lora_layer = lora_layer

    def forward(self, x):
        if self.lora_layer is None:
            return super().forward(x)
        else:
            return super().forward(x) + self.lora_layer(x)
