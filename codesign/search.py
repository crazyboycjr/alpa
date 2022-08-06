"""Search the model specifications."""
from tokenize import Token
from typing import List, Sequence, Callable
import unittest
import copy
import random
import itertools

from tqdm import tqdm
import jax.tree_util

from zhen import TokenMixer
from model import ModelSpec, create_model, get_token_mixer

global pbar
BAR_FORMAT = "{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}"


def explode(placed: List[TokenMixer], last_op: int, num_ops: int,
            tokens: Sequence[TokenMixer]):
    """Enumerate all the combinations of putting `num_ops` tokens, selected from `tokens`."""
    # print(placed, last_op, num_ops, tokens)
    if len(placed) == num_ops:
        yield placed
        return
    for op in range(last_op, len(tokens)):
        # print('op', op)
        placed.append(tokens[op])
        for sol in explode(placed, op, num_ops, tokens):
            yield sol
        placed.pop()


def dfs(ops: List[List[TokenMixer]], cur_layer: int, placed_ops: int,
        tokens: Sequence[TokenMixer], num_layers: int, total_ops: int,
        result: List[List[List[TokenMixer]]]):
    global pbar

    if total_ops - placed_ops < num_layers - cur_layer:
        return

    if cur_layer + 1 == num_layers:
        for layer_ops in explode([], 0, total_ops - placed_ops, tokens):
            ops[cur_layer] = layer_ops
            result.append(copy.deepcopy(ops))
            pbar.update(1)
            # print(ops)
        return

    for i in range(1, total_ops - placed_ops + 1):
        for layer_ops in explode([], 0, i, tokens):
            ops[cur_layer] = layer_ops
            dfs(ops, cur_layer + 1, placed_ops + i, tokens, num_layers,
                total_ops, result)


def base_model_spec():
    return {
        'num_features':
            512,
        'emb_dim':
            160,
        'output_per_emb':
            50,
        'num_zhen_layers':
            4,
        'tokens': [
            get_token_mixer(t)
            for t in ["ATTENTION", "LINEAR", "ATTENTION", "DOT"]
        ],
    }


def search_model_fix_num_layers(tokens: Sequence[TokenMixer], num_layers: int,
                                total_ops: int) -> List[ModelSpec]:
    ops = [[] for _ in range(num_layers)]
    result = []
    dfs(ops, 0, 0, tokens, num_layers, total_ops, result)

    models = []

    base_spec = base_model_spec()

    for ops in result:
        # construct a model_spec
        model_spec = copy.copy(base_spec)
        model_spec.update({
            'num_zhen_layers': len(ops),
            'tokens': copy.copy(ops),
        })
        models.append(model_spec)
    return models


def search_models(tokens: Sequence[TokenMixer], max_layers: int,
                  total_ops: int) -> List[ModelSpec]:
    models = []
    for num_layers in range(1, max_layers + 1):
        models_tmp = search_model_fix_num_layers(tokens, num_layers, total_ops)
        models += models_tmp
    return models


def count_candidates(tokens: Sequence[TokenMixer], max_layers: int,
                     total_ops: int) -> int:
    import numpy as np
    num_tokens = len(tokens)
    f = np.ones((total_ops + 1, num_tokens + 1), np.int64)
    g = np.ones((total_ops + 1, num_tokens + 1), np.int64)
    for i in range(1, num_tokens + 1):
        g[1][i] = i
    for i in range(2, total_ops + 1):
        for j in range(1, num_tokens + 1):
            f[i][j] = g[i - 1][num_tokens] - g[i - 1][j - 1]
            g[i][j] = g[i][j - 1] + f[i][j]

    def dfs_count(a: List[int], k: int, placed_ops: int, num_layers: int,
                  total_ops: int, num_tokens: int) -> int:
        if total_ops - placed_ops < num_layers - k:
            return 0

        if k + 1 == num_layers:
            a[k] = total_ops - placed_ops
            return np.product([g[x][num_tokens] for x in a])

        s = 0
        for i in range(1, total_ops - placed_ops + 1):
            a[k] = i
            s += dfs_count(a, k + 1, placed_ops + i, num_layers, total_ops,
                           num_tokens)
        return s

    s = 0
    for num_layers in range(1, max_layers + 1):
        a = [0 for _ in range(num_layers)]
        s += dfs_count(a, 0, 0, num_layers, total_ops, num_tokens)
    return s


def count_params(model_spec: ModelSpec) -> int:
    m = create_model(model_spec)
    total_params = sum(p.numel() for p in m.parameters())
    return total_params


class ModelFilter(object):

    def __init__(self, models: Sequence[ModelSpec],
                 filters: Sequence[Callable[[ModelSpec], bool]]):
        self.models = models
        self.filters = filters

    def iter(self):
        for model_spec in self.models:
            if all(f(model_spec) for f in self.filters):
                print('find a solution:', model_spec)
                yield model_spec


def fixed_num_layer(num_layers: int) -> Callable[[ModelSpec], bool]:

    def inner(model_spec: ModelSpec) -> bool:
        return num_layers == model_spec['num_zhen_layers']

    return inner


def similar_parameter_count(base_count: int,
                            diff_bound: float) -> Callable[[ModelSpec], bool]:
    """Filter models whose difference in parameter count with the default model is within diff_bound."""

    # abs(a - b) / max(a, b) < 0.1
    assert diff_bound >= 0.0 and diff_bound < 1.0

    def inner(model_spec: ModelSpec) -> bool:
        total_params = count_params(model_spec)
        a = total_params
        b = base_count
        diff = (a - b) / max(a, b)
        if abs(diff) < diff_bound:
            print("model_spec: {}, total_params {}, diff: {}".format(
                model_spec, total_params, diff))
            return True
        else:
            print("model_spec: {}, total_params {}, diff: {}, skipped".format(
                model_spec, total_params, diff))
            return False

    return inner


def uniform_token_mixer(token: TokenMixer) -> Callable[[ModelSpec], bool]:
    """Filter the models whose token mixer are all `token`"""

    def inner(model_spec: ModelSpec) -> bool:
        return all(t == token
                   for t in jax.tree_util.tree_flatten(model_spec['tokens'])[0])

    return inner

def add_dot_models(ret: List[ModelSpec], models: Sequence[ModelSpec]):
    default_total_params = count_params(base_model_spec())
    print("model_spec: {}, total_params {}".format(base_model_spec(),
                                                   default_total_params))

    ret.append(base_model_spec())

    model_filter = ModelFilter(models, [
        uniform_token_mixer(TokenMixer.DOT),
        similar_parameter_count(default_total_params, 0.1)
    ])
    for model_spec in itertools.islice(model_filter.iter(), 10):
        ret.append(model_spec)

    model_filter = ModelFilter(models, [
        fixed_num_layer(1),
        uniform_token_mixer(TokenMixer.DOT),
        similar_parameter_count(default_total_params, 0.1),
    ])
    for model_spec in itertools.islice(model_filter.iter(), 2):
        ret.append(model_spec)

    model_filter = ModelFilter(models, [
        fixed_num_layer(2),
        uniform_token_mixer(TokenMixer.DOT),
        similar_parameter_count(default_total_params, 0.1),
    ])
    for model_spec in itertools.islice(model_filter.iter(), 2):
        ret.append(model_spec)

    model_filter = ModelFilter(models, [
        fixed_num_layer(3),
        uniform_token_mixer(TokenMixer.DOT),
        similar_parameter_count(default_total_params, 0.1),
    ])
    for model_spec in itertools.islice(model_filter.iter(), 2):
        ret.append(model_spec)


def add_attention_models(ret: List[ModelSpec], models: Sequence[ModelSpec]):
    # try model consists of pure attention layers
    base_attention_model = base_model_spec()
    base_attention_model["tokens"] = [
        TokenMixer.ATTENTION
        for _ in range(base_attention_model["num_zhen_layers"])
    ]
    default_total_params = count_params(base_attention_model)
    print("model_spec: {}, total_params {}".format(base_attention_model,
                                                   default_total_params))

    ret.append(base_attention_model)

    model_filter = ModelFilter(models, [
        uniform_token_mixer(TokenMixer.ATTENTION),
        similar_parameter_count(default_total_params, 0.1)
    ])
    for model_spec in itertools.islice(model_filter.iter(), 10):
        ret.append(model_spec)


def add_linear_models(ret: List[ModelSpec], models: Sequence[ModelSpec]):
    # try model consists of pure linear layers
    base_linear_model = base_model_spec()
    base_linear_model["tokens"] = [
        TokenMixer.LINEAR
        for _ in range(base_linear_model["num_zhen_layers"])
    ]
    default_total_params = count_params(base_linear_model)
    print("model_spec: {}, total_params {}".format(base_linear_model,
                                                   default_total_params))

    ret.append(base_linear_model)

    model_filter = ModelFilter(models, [
        uniform_token_mixer(TokenMixer.LINEAR),
        similar_parameter_count(default_total_params, 0.1)
    ])
    for model_spec in itertools.islice(model_filter.iter(), 10):
        ret.append(model_spec)



def search_model() -> List[ModelSpec]:
    # TODO(cjr): Remove convolution operator since alpa does not support it perfectly
    tokens = [
        TokenMixer.DOT,
        TokenMixer.LINEAR,
        TokenMixer.ATTENTION,
        # TokenMixer.CONVOLUTION
    ]
    num_candidates = count_candidates(tokens, 4, 12)
    print('num_candidates', num_candidates)
    global pbar
    pbar = tqdm(total=num_candidates, bar_format=BAR_FORMAT)

    models = search_models(tokens, 4, 12)
    print(len(models))
    assert num_candidates == len(models)

    pbar.close()

    # take the first 200 models candidates
    random.shuffle(models)

    ret = []

    add_dot_models(ret, models)
    add_attention_models(ret, models)
    add_linear_models(ret, models)

    return ret


class TestSearch(unittest.TestCase):

    def setUp(self):
        pass

    def test1(self):
        tokens = [
            TokenMixer.DOT, TokenMixer.LINEAR, TokenMixer.ATTENTION,
            TokenMixer.CONVOLUTION
        ]
        num_candidates = count_candidates(tokens, 4, 12)
        print('num_candidates', num_candidates)
        global pbar
        pbar = tqdm(total=num_candidates, bar_format=BAR_FORMAT)

        models = search_models(tokens, 4, 12)
        print(len(models))
        assert num_candidates == len(models)

        pbar.close()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestSearch("test_zhen_homogeneous"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())