"""Search the model specifications."""
from tokenize import Token
from typing import List, Sequence

from zhen import TokenMixer
from model import ModelSpec


def explode(placed: List[TokenMixer], last_op: int, num_ops: int, tokens: Sequence[TokenMixer]):
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
    if total_ops - placed_ops < num_layers - cur_layer:
        return

    if cur_layer + 1 == num_layers:
        for layer_ops in explode([], 0, total_ops - placed_ops, tokens):
            ops[cur_layer] = layer_ops
            result.append(ops)
            print(ops)
        return

    for i in range(1, total_ops - placed_ops + 1):
        for layer_ops in explode([], 0, i, tokens):
            ops[cur_layer] = layer_ops
            dfs(ops, cur_layer + 1, placed_ops + i, tokens, num_layers, total_ops, result)


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
            'tokens': ops,
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


def test():
    tokens = [
        TokenMixer.DOT, TokenMixer.LINEAR, TokenMixer.ATTENTION,
        TokenMixer.CONVOLUTION
    ]
    models = search_models(tokens, 4, 12)
    print(len(models))


# #include <cstdio>
#
# int a[20];
# int b[20];
# int f[20][20];
# int g[20][20];
# int num_tokens = 4;
# int search_space = 0;
#
# void dfs(int k, int ops, int num_layers, int total_ops) {
#   if (total_ops - ops < num_layers - k) {
#     return;
#   }
#
#   if (k + 1 == num_layers) {
#     a[k] = total_ops - ops;
#     int s = 1;
#     for (int i = 0; i < num_layers; i++) {
#       printf("%d, ", a[i]);
#       s *= g[a[i]][num_tokens];
#     }
#     search_space += s;
#     printf("b = %d\n", s);
#     return;
#   }
#
#   for (int i = 1; i <= total_ops - ops; i++) {
#     a[k] = i;
#     dfs(k + 1, ops + i, num_layers, total_ops);
#   }
# }
#
# int main() {
#   int total_ops = 12;
#   for (int i = 1; i <= num_tokens; i++) {
#     f[1][i] = 1;
#     g[1][i] = i;
#   }
#   for (int i = 2; i <= total_ops; i++) {
#     for (int j = 1; j <= num_tokens; j++) {
#       f[i][j] = g[i - 1][num_tokens] - g[i - 1][j - 1];
#       g[i][j] = g[i][j - 1] + f[i][j];
#       printf("%d %d %d\n", i, j, f[i][j]);
#     }
#   }
#
#   dfs(0, 0, 4, 12);
#
#   printf("%d\n", search_space);
# }