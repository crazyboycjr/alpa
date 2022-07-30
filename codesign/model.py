from typing import Dict, Any

import torch
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
    return f'{num_zhen_layers}, {tokens}, {num_features}, {emb_dim}, {output_per_emb}'