from typing import Dict, Any
import copy

import torch
import jax.tree_util
from serde.toml import to_toml

from zhen import ZHENCollection, TokenMixer

ModelSpec = Dict[str, Any]


def get_token_mixer(t: str) -> TokenMixer:
    if t == 'ATTENTION':
        return TokenMixer.ATTENTION
    if t == 'LINEAR':
        return TokenMixer.LINEAR
    if t == 'DOT':
        return TokenMixer.DOT
    if t == 'CONVOLUTION':
        return TokenMixer.CONVOLUTION
    raise NotImplementedError(f"Unknown token mixer {t}")


def token_mixer_to_str(t: TokenMixer) -> str:
    if t == TokenMixer.DOT:
        return 'DOT'
    if t == TokenMixer.LINEAR:
        return 'LINEAR'
    if t == TokenMixer.ATTENTION:
        return 'ATTENTION'
    if t == TokenMixer.CONVOLUTION:
        return 'CONVOLUTION'
    raise NotImplementedError(f"Unknown token mixer {t}, {type(t)}")


def create_model(model_spec: ModelSpec) -> torch.nn.Module:
    num_features = model_spec['num_features']
    emb_dim = model_spec['emb_dim']
    output_per_emb = model_spec['output_per_emb']
    num_zhen_layers = model_spec['num_zhen_layers']
    tokens = model_spec['tokens']
    return ZHENCollection(num_zhen_layers, emb_dim, tokens, num_features,
                          output_per_emb)

def to_sql_values(model_spec: ModelSpec) -> str:
    num_features = model_spec['num_features']
    emb_dim = model_spec['emb_dim']
    output_per_emb = model_spec['output_per_emb']
    num_zhen_layers = model_spec['num_zhen_layers']
    tokens = model_spec['tokens']
    return f'{num_zhen_layers}, "{tokens}", {num_features}, {emb_dim}, {output_per_emb}'


def dump_to_toml(model_spec: ModelSpec) -> str:
    d = copy.deepcopy(model_spec)
    d["tokens"] = jax.tree_util.tree_map(token_mixer_to_str, d["tokens"])
    return to_toml(d)