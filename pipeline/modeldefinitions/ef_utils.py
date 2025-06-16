# Licensed to the GluonNLP team under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Create a registry."""

from typing import Optional, List
import json
from json import JSONDecodeError


class Registry:
    """Create the registry that will map name to object. This facilitates the users to create
    custom registry.

    Parameters
    ----------
    name
        The name of the registry

    Examples
    --------

    >>> from earthformer.utils.registry import Registry
    >>> # Create a registry
    >>> MODEL_REGISTRY = Registry('MODEL')
    >>>
    >>> # To register a class/function with decorator
    >>> @MODEL_REGISTRY.register()
...     class MyModel:
...         pass
    >>> @MODEL_REGISTRY.register()
...     def my_model():
...         return
    >>>
    >>> # To register a class object with decorator and provide nickname:
    >>> @MODEL_REGISTRY.register('test_class')
...     class MyModelWithNickName:
...         pass
    >>> @MODEL_REGISTRY.register('test_function')
...     def my_model_with_nick_name():
...         return
    >>>
    >>> # To register a class/function object by function call
...     class MyModel2:
...         pass
    >>> MODEL_REGISTRY.register(MyModel2)
    >>> # To register with a given name
    >>> MODEL_REGISTRY.register('my_model2', MyModel2)
    >>> # To list all the registered objects:
    >>> MODEL_REGISTRY.list_keys()

['MyModel', 'my_model', 'test_class', 'test_function', 'MyModel2', 'my_model2']

    >>> # To get the registered object/class
    >>> MODEL_REGISTRY.get('test_class')

__main__.MyModelWithNickName

    """

    def __init__(self, name: str) -> None:
        self._name: str = name
        self._obj_map: dict[str, object] = dict()

    def _do_register(self, name: str, obj: object) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, *args):
        """
        Register the given object under either the nickname or `obj.__name__`. It can be used as
         either a decorator or not. See docstring of this class for usage.
        """
        if len(args) == 2:
            # Register an object with nick name by function call
            nickname, obj = args
            self._do_register(nickname, obj)
        elif len(args) == 1:
            if isinstance(args[0], str):
                # Register an object with nick name by decorator
                nickname = args[0]
                def deco(func_or_class: object) -> object:
                    self._do_register(nickname, func_or_class)
                    return func_or_class
                return deco
            else:
                # Register an object by function call
                self._do_register(args[0].__name__, args[0])
        elif len(args) == 0:
            # Register an object by decorator
            def deco(func_or_class: object) -> object:
                self._do_register(func_or_class.__name__, func_or_class)
                return func_or_class
            return deco
        else:
            raise ValueError('Do not support the usage!')

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name
                )
            )
        return ret

    def list_keys(self) -> List:
        return list(self._obj_map.keys())

    def __repr__(self) -> str:
        s = '{name}(keys={keys})'.format(name=self._name,
                                         keys=self.list_keys())
        return s

    def create(self, name: str, *args, **kwargs) -> object:
        """Create the class object with the given args and kwargs

        Parameters
        ----------
        name
            The name in the registry
        args
        kwargs

        Returns
        -------
        ret
            The created object
        """
        obj = self.get(name)
        try:
            return obj(*args, **kwargs)
        except Exception as exp:
            print('Cannot create name="{}" --> {} with the provided arguments!\n'
                  '   args={},\n'
                  '   kwargs={},\n'
                  .format(name, obj, args, kwargs))
            raise exp

    def create_with_json(self, name: str, json_str: str):
        """

        Parameters
        ----------
        name
        json_str

        Returns
        -------

        """
        try:
            args = json.loads(json_str)
        except JSONDecodeError:
            raise ValueError('Unable to decode the json string: json_str="{}"'
                             .format(json_str))
        if isinstance(args, (list, tuple)):
            return self.create(name, *args)
        elif isinstance(args, dict):
            return self.create(name, **args)
        else:
            raise NotImplementedError('The format of json string is not supported! We only support '
                                      'list/dict. json_str="{}".'
                                      .format(json_str))

"""Patterns for cuboid self-attention / cross attention"""

import functools

CuboidSelfAttentionPatterns = Registry('CuboidSelfAttentionPattern')
CuboidCrossAttentionPatterns = Registry('CuboidCrossAttentionPatterns')

# basic patterns

def full_attention(input_shape):
    T, H, W, _ = input_shape
    cuboid_size = [(T, H, W)]
    strategy = [('l', 'l', 'l')]
    shift_size = [(0, 0, 0)]
    return cuboid_size, strategy, shift_size

def self_axial(input_shape):
    """Axial attention proposed in https://arxiv.org/abs/1912.12180

    Parameters
    ----------
    input_shape
        T, H, W

    Returns
    -------
    cuboid_size
    strategy
    shift_size
    """
    T, H, W, _ = input_shape
    cuboid_size = [(T, 1, 1), (1, H, 1), (1, 1, W)]
    strategy = [('l', 'l', 'l'), ('l', 'l', 'l'), ('l', 'l', 'l')]
    shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size

def self_video_swin(input_shape, P=2, M=4):
    """Adopt the strategy in Video SwinTransformer https://arxiv.org/pdf/2106.13230.pdf"""
    T, H, W, _ = input_shape
    P = min(P, T)
    M = min(M, H, W)
    cuboid_size = [(P, M, M), (P, M, M)]
    strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
    shift_size = [(0, 0, 0), (P // 2, M // 2, M // 2)]

    return cuboid_size, strategy, shift_size

def self_divided_space_time(input_shape):
    T, H, W, _ = input_shape
    cuboid_size = [(T, 1, 1), (1, H, W)]
    strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
    shift_size = [(0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size

# basic patterns
CuboidSelfAttentionPatterns.register('full', full_attention)
CuboidSelfAttentionPatterns.register('axial', self_axial)
CuboidSelfAttentionPatterns.register('video_swin', self_video_swin)
CuboidSelfAttentionPatterns.register('divided_st', self_divided_space_time)
# video_swin_PxM
for p in [1, 2, 4, 8, 10]:
    for m in [1, 2, 4, 8, 16, 32]:
        CuboidSelfAttentionPatterns.register(
            f'video_swin_{p}x{m}',
            functools.partial(self_video_swin,
                              P=p, M=m))

# our proposals
def self_spatial_lg_v1(input_shape, M=4):
    T, H, W, _ = input_shape

    if H <= M and W <= M:
        cuboid_size = [(T, 1, 1), (1, H, W)]
        strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
        shift_size = [(0, 0, 0), (0, 0, 0)]
    else:
        cuboid_size = [(T, 1, 1), (1, M, M), (1, M, M)]
        strategy = [('l', 'l', 'l'), ('l', 'l', 'l'), ('d', 'd', 'd')]
        shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size


# Following are our proposed new patterns based on the CuboidSelfAttention design.
CuboidSelfAttentionPatterns.register('spatial_lg_v1', self_spatial_lg_v1)
# spatial_lg
for m in [1, 2, 4, 8, 16, 32]:
    CuboidSelfAttentionPatterns.register(
        f'spatial_lg_{m}',
        functools.partial(self_spatial_lg_v1,
                          M=m))

def self_axial_space_dilate_K(input_shape, K=2):
    T, H, W, _ = input_shape
    K = min(K, H, W)
    cuboid_size = [(T, 1, 1),
                   (1, H // K, 1), (1, H // K, 1),
                   (1, 1, W // K), (1, 1, W // K)]
    strategy = [('l', 'l', 'l'),
                ('d', 'd', 'd'), ('l', 'l', 'l'),
                ('d', 'd', 'd'), ('l', 'l', 'l'),]
    shift_size = [(0, 0, 0),
                  (0, 0, 0), (0, 0, 0),
                  (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size
for k in [2, 4, 8]:
    CuboidSelfAttentionPatterns.register(
        f'axial_space_dilate_{k}',
        functools.partial(self_axial_space_dilate_K,
                          K=k))


def cross_KxK(mem_shape, K):
    """

    Parameters
    ----------
    mem_shape
    K

    Returns
    -------
    cuboid_hw
    shift_hw
    strategy
    n_temporal
    """
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K)]
    shift_hw = [(0, 0)]
    strategy = [('l', 'l', 'l')]
    n_temporal = [1]
    return cuboid_hw, shift_hw, strategy, n_temporal

def cross_KxK_lg(mem_shape, K):
    """

    Parameters
    ----------
    mem_shape
    K

    Returns
    -------
    cuboid_hw
    shift_hw
    strategy
    n_temporal
    """
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K), (K, K)]
    shift_hw = [(0, 0), (0, 0)]
    strategy = [('l', 'l', 'l'), ('d', 'd', 'd')]
    n_temporal = [1, 1]
    return cuboid_hw, shift_hw, strategy, n_temporal

def cross_KxK_heter(mem_shape, K):
    """

    Parameters
    ----------
    mem_shape
    K

    Returns
    -------
    cuboid_hw
    shift_hw
    strategy
    n_temporal
    """
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K), (K, K), (K, K)]
    shift_hw = [(0, 0), (0, 0), (K // 2, K // 2)]
    strategy = [('l', 'l', 'l'), ('d', 'd', 'd'), ('l', 'l', 'l')]
    n_temporal = [1, 1, 1]
    return cuboid_hw, shift_hw, strategy, n_temporal

# # Our proposed CuboidCrossAttention patterns.
for k in [1, 2, 4, 8]:
    CuboidCrossAttentionPatterns.register(f'cross_{k}x{k}', functools.partial(cross_KxK, K=k))
    CuboidCrossAttentionPatterns.register(f'cross_{k}x{k}_lg', functools.partial(cross_KxK_lg, K=k))
    CuboidCrossAttentionPatterns.register(f'cross_{k}x{k}_heter', functools.partial(cross_KxK_heter, K=k))


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def round_to(dat, c):
    return dat + (dat - dat % c) % c

def get_activation(act, inplace=False, **kwargs):
    """

    Parameters
    ----------
    act
        Name of the activation
    inplace
        Whether to perform inplace activation

    Returns
    -------
    activation_layer
        The activation
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            negative_slope = kwargs.get("negative_slope", 0.1)
            return nn.LeakyReLU(negative_slope, inplace=inplace)
        elif act == 'identity':
            return nn.Identity()
        elif act == 'elu':
            return nn.ELU(inplace=inplace)
        elif act == 'gelu':
            return nn.GELU()
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'softrelu' or act == 'softplus':
            return nn.Softplus()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError('act="{}" is not supported. '
                                      'Try to include it if you can find that in '
                                      'https://pytorch.org/docs/stable/nn.html'.format(act))
    else:
        return act

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """Root Mean Square Layer Normalization proposed in "[NeurIPS2019] Root Mean Square Layer Normalization"

        Parameters
        ----------
        d
            model size
        p
            partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        eps
            epsilon value, default 1e-8
        bias
            whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

def get_norm_layer(normalization: str = 'layer_norm',
                   axis: int = -1,
                   epsilon: float = 1e-5,
                   in_channels: int = 0, **kwargs):
    """Get the normalization layer based on the provided type

    Parameters
    ----------
    normalization
        The type of the layer normalization from ['layer_norm']
    axis
        The axis to normalize the
    epsilon
        The epsilon of the normalization layer
    in_channels
        Input channel

    Returns
    -------
    norm_layer
        The layer normalization layer
    """
    if isinstance(normalization, str):
        if normalization == 'layer_norm':
            assert in_channels > 0
            assert axis == -1
            norm_layer = nn.LayerNorm(normalized_shape=in_channels, eps=epsilon, **kwargs)
        elif normalization == 'rms_norm':
            assert axis == -1
            norm_layer = RMSNorm(d=in_channels, eps=epsilon, **kwargs)
        else:
            raise NotImplementedError('normalization={} is not supported'.format(normalization))
        return norm_layer
    elif normalization is None:
        return nn.Identity()
    else:
        raise NotImplementedError('The type of normalization must be str')


def _generalize_padding(x, pad_t, pad_h, pad_w, padding_type, t_pad_left=False):
    """

    Parameters
    ----------
    x
        Shape (B, T, H, W, C)
    pad_t
    pad_h
    pad_w
    padding_type
    t_pad_left

    Returns
    -------
    out
        The result after padding the x. Shape will be (B, T + pad_t, H + pad_h, W + pad_w, C)
    """
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    assert padding_type in ['zeros', 'ignore', 'nearest']
    B, T, H, W, C = x.shape

    if padding_type == 'nearest':
        return F.interpolate(x.permute(0, 4, 1, 2, 3), size=(T + pad_t, H + pad_h, W + pad_w)).permute(0, 2, 3, 4, 1)
    else:
        if t_pad_left:
            return F.pad(x, (0, 0, 0, pad_w, 0, pad_h, pad_t, 0))
        else:
            return F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))


def _generalize_unpadding(x, pad_t, pad_h, pad_w, padding_type):
    assert padding_type in['zeros', 'ignore', 'nearest']
    B, T, H, W, C = x.shape
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    if padding_type == 'nearest':
        return F.interpolate(x.permute(0, 4, 1, 2, 3), size=(T - pad_t, H - pad_h, W - pad_w)).permute(0, 2, 3, 4, 1)
    else:
        return x[:, :(T - pad_t), :(H - pad_h), :(W - pad_w), :].contiguous()

def apply_initialization(m,
                         linear_mode="0",
                         conv_mode="0",
                         norm_mode="0",
                         embed_mode="0"):
    if isinstance(m, nn.Linear):

        if linear_mode in ("0", ):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_in', nonlinearity="linear")
        elif linear_mode in ("1", ):
            nn.init.kaiming_normal_(m.weight,
                                    a=0.1,
                                    mode='fan_out',
                                    nonlinearity="leaky_relu")
        else:
            raise NotImplementedError
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        if conv_mode in ("0", ):
            nn.init.kaiming_normal_(m.weight,
                                    a=0.1,
                                    mode='fan_out',
                                    nonlinearity="leaky_relu")
        else:
            raise NotImplementedError
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if norm_mode in ("0", ):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError
    elif isinstance(m, nn.GroupNorm):
        if norm_mode in ("0", ):
            if m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError
    # # pos_embed already initialized when created
    elif isinstance(m, nn.Embedding):
        if embed_mode in ("0", ):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
        else:
            raise NotImplementedError
    else:
        pass
