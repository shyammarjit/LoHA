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
import torch, math

def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    In LoRA with Kroneckor Product, first value is a value for weight scale.
    secon value is a value for weight.

    Becuase of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 1, 127   127 -> 1, 127    127 -> 1, 127   127 -> 1, 127   127 -> 1, 127
    128 -> 8, 16    128 -> 2, 64     128 -> 4, 32    128 -> 8, 16    128 -> 8, 16
    250 -> 10, 25   250 -> 2, 125    250 -> 2, 125   250 -> 5, 50    250 -> 10, 25
    360 -> 8, 45    360 -> 2, 180    360 -> 4, 90    360 -> 8, 45    360 -> 12, 30
    512 -> 16, 32   512 -> 2, 256    512 -> 4, 128   512 -> 8, 64    512 -> 16, 32
    1024 -> 32, 32  1024 -> 2, 512   1024 -> 4, 256  1024 -> 8, 128  1024 -> 16, 64

    # Note: This code is taken from https://github.com/KohakuBlueleaf/LyCORIS/blob/main/lycoris/modules/lokr.py
    """

    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n

# def kronecker(A, B):
#     return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def make_kron(w1, w2):
    # print(w1.shape, w2.shape)
    # This function performs kronecker decomposition
    # Note: This code is taken from https://github.com/KohakuBlueleaf/LyCORIS/blob/main/lycoris/modules/lokr.py
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)

    return rebuild

class LoKr(nn.Module):
    # Note: This code is taken from https://github.com/KohakuBlueleaf/LyCORIS/blob/main/lycoris/modules/lokr.py
    def __init__(self, in_features, out_features, network_alpha=None, factor=-1, lora_rank=4, decompose_both=True, 
        device=None, 
        dtype=None,
        use_scalar=False,      
    ):
        # print(factor)
        """
        decompose_both: If you want low rank decomposition (LoRA) for upper and lower matrices.
        out_dim = out_l * in_m
        in_dim = out_k * in_n
        """
        super().__init__()
        self.use_w1 = False
        self.use_w2 = False
        self.in_features = in_features
        self.out_features = out_features

        # get the Kronecker factors for upper and lower matrix
        in_m, in_n = factorization(in_features, factor) 
        out_l, out_k = factorization(out_features, factor)
        self.a1, self.b1 = out_l, out_k
        self.a2, self.b2 = in_m, in_n
        
        
        # W1: smaller part
        if decompose_both and lora_rank < max(out_l, in_m) / 2:
            self.lokr_w1_a = nn.Linear(lora_rank, out_l, bias=False, device=device, dtype=dtype)
            self.lokr_w1_b = nn.Linear(in_m, lora_rank, bias=False, device=device, dtype=dtype)
        else:
            self.use_w1 = True
            self.lokr_w1 = nn.Linear(in_m, out_l, bias=False, device=device, dtype=dtype)  # a*c, 1-mode
        
        # W2: bigger part, weight and LoRA. [b, dim] x [dim, d]
        if lora_rank < max(out_k, in_n) / 2:
            self.lokr_w2_a = nn.Linear(lora_rank, out_k, bias=False, device=device, dtype=dtype)
            self.lokr_w2_b = nn.Linear(in_n, lora_rank, bias=False, device=device, dtype=dtype)
            # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
        else:
            self.use_w2 = True
            self.lokr_w2 = nn.Linear(in_n, out_k, bias=False, device=device, dtype=dtype)

        if self.use_w2:
            if use_scalar:
                nn.init.kaiming_uniform_(self.lokr_w2.weight, a=math.sqrt(5))
            else:
                nn.init.constant_(self.lokr_w2.weight, 0)
        else:
            nn.init.kaiming_uniform_(self.lokr_w2_a.weight, a=math.sqrt(5))
            if use_scalar: # basically to update the value of `b` in kaiming_uniform_
                nn.init.kaiming_uniform_(self.lokr_w2_b.weight, a=math.sqrt(5))
            else:
                nn.init.constant_(self.lokr_w2_b.weight, 0)

        if self.use_w1:
            nn.init.kaiming_uniform_(self.lokr_w1.weight, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.lokr_w1_a.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lokr_w1_b.weight, a=math.sqrt(5))
        
    def forward(self, hidden_states):
        # get the down matrix
        down = self.lokr_w1.weight if self.use_w1 else self.lokr_w1_a.weight @ self.lokr_w1_b.weight
        # get the up matrix
        up = self.lokr_w2.weight if self.use_w2 else self.lokr_w2_a.weight @ self.lokr_w2_b.weight
        weight = make_kron(down, up)
        out = hidden_states @ weight.T
        return out

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None, lphm=None):
        super().__init__()
        # print(kamal)
        # exit()
        self.lphm = lphm
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        
        if lphm:
            # self.down_in = nn.Parameter(torch.FloatTensor(in_features, 1).to(dtype).to(device), requires_grad=True)
            self.down_in = nn.Linear(1, in_features, bias=False, device=device, dtype=dtype)
            # self.down_out = nn.Parameter(torch.FloatTensor(1, rank).to(dtype).to(device), requires_grad=True)
            self.down_out = nn.Linear(rank, 1, bias=False, device=device, dtype=dtype)

            # self.up_in = nn.Parameter(torch.FloatTensor(rank, 1).to(dtype).to(device), requires_grad=True)
            self.up_in = nn.Linear(1, rank, bias=False, device=device, dtype=dtype)
            # self.up_out = nn.Parameter(torch.FloatTensor(1, out_features).to(dtype).to(device), requires_grad=True)
            self.up_out = nn.Linear(out_features, 1, bias=False, device=device, dtype=dtype)

            nn.init.normal_(self.down_in.weight, std=1 / rank)
            nn.init.normal_(self.down_out.weight, std=1 / rank)
            nn.init.zeros_(self.up_in.weight)
            nn.init.zeros_(self.up_out.weight)
            # self.down = torch.kron(self.down_in.weight, self.down_out.weight)#.to(dtype).to(device)
            # self.up = torch.kron(self.up_in.weight, self.up_out.weight)#.to(dtype).to(device)
            
        else: 
            self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
            self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
            
            nn.init.normal_(self.down.weight, std=1 / rank)
            nn.init.zeros_(self.up.weight)

        
        

    def forward(self, hidden_states):
        # print('in')
        # print(self.down_in.weight.device)
        # exit()
        orig_dtype = hidden_states.dtype

        if self.lphm:
            dtype = self.down_in.weight.dtype
            device = self.down_in.weight.device
            # print(hidden_states.shape, self.down.shape)
            # exit()
            self.down = torch.kron(self.down_in.weight, self.down_out.weight)#.to(dtype).to(device)
            self.up = torch.kron(self.up_in.weight, self.up_out.weight)#.to(dtype).to(device)
            down_hidden_states = hidden_states.to(dtype)@self.down.to(device)
            up_hidden_states = down_hidden_states@self.up.to(device)
        else:
            dtype = self.down.weight.dtype
            down_hidden_states = self.down(hidden_states.to(dtype))
            up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRAConv2dLayer(nn.Module):
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


class LoRACompatibleConv(nn.Conv2d):
    """
    A convolutional layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRAConv2dLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRAConv2dLayer]):
        self.lora_layer = lora_layer

    def forward(self, x):
        """It's None."""
        if self.lora_layer is None:
            # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
            # see: https://github.com/huggingface/diffusers/pull/4315
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return super().forward(x) + self.lora_layer(x)


class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, adapter_low_rank=None, **kwargs):
        super().__init__(*args, **kwargs)
        # print('shyam', args, kwargs)
        # print(kamal)
        self.lora_layer = lora_layer
        self.args = args
        self.kwargs = kwargs
    
    def get_config(self):
        return self.args, self.kwargs

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    def forward(self, x):
        if self.lora_layer is None:
            return super().forward(x)
        else:
            return super().forward(x) + self.lora_layer(x)
