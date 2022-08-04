"""Search the model specifications."""
import unittest
import copy
import random
from typing import List, Sequence

from tqdm import tqdm

from zhen import TokenMixer
from model import ModelSpec

global pbar
BAR_FORMAT="{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}"

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


def search_model_fix_num_layers(tokens: Sequence[TokenMixer], num_layers: int,
                                total_ops: int) -> List[ModelSpec]:
    ops = [[] for _ in range(num_layers)]
    result = []
    dfs(ops, 0, 0, tokens, num_layers, total_ops, result)

    models = []
    for ops in result:
        # construct a model_spec
        model_spec = {
            'num_features': 512,
            'emb_dim': 160,
            'output_per_emb': 20,
            'num_zhen_layers': num_layers,
            'tokens': copy.copy(ops),
        }
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


def search_model() -> List[ModelSpec]:
    # TODO(cjr): Remove convolution operator since alpa does not support it perfectly
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

    # take the first 200 models candidates
    random.shuffle(models)
    return models[:200]


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestSearch("test_zhen_homogeneous"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())