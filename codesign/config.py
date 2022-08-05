from dataclasses import dataclass
from typing import Tuple, List

import jax.tree_util
from serde import serde
from serde.toml import from_toml

from cluster import ClusterSpec
from training_spec import TrainingSpec
from parallel import ParallelSpec
from model import ModelSpec, get_token_mixer


@serde
@dataclass
class Config(object):
    training_spec: TrainingSpec
    model: List[ModelSpec]
    cluster_spec: List[ClusterSpec]
    parallel_spec: List[ParallelSpec]

    @staticmethod
    def load(path: str):
        content = open(path, 'r').read()
        return from_toml(Config, content)

    def generate_cluster_specs(self):
        for c in self.cluster_spec:
            yield c

    def generate_models(
            self, cluster_spec: ClusterSpec) -> Tuple[ModelSpec, TrainingSpec]:
        import torch
        import alpa.torch.optim as torchoptim

        for c in self.model:
            tokens = jax.tree_util.tree_map(get_token_mixer, c["tokens"])
            print(tokens)
            model_spec = c.copy()
            model_spec['tokens'] = tokens

            num_gpus = cluster_spec.num_gpus()

            avg_batch_size_per_device = self.training_spec.avg_batch_size_per_device
            num_iters = self.training_spec.num_iters
            global_batch_size = avg_batch_size_per_device * num_gpus
            loss_func = lambda *args, **kwargs: torch.nn.functional.mse_loss(
                *args, **kwargs)
            optim_gen = torchoptim.adam(lr=1e-3)

            training_spec = TrainingSpec(avg_batch_size_per_device, num_iters,
                                         global_batch_size, loss_func,
                                         optim_gen)

            yield model_spec, training_spec

    def generate_parallel_specs(self, cluster_spec: ClusterSpec):
        num_gpus = cluster_spec.num_gpus()

        for p in self.parallel_spec:
            # explode by this dimension
            for num_micro_batches in p["num_micro_batches"]:
                ret = p.copy()
                ret["num_micro_batches"] = num_micro_batches

                # high-level parallelism specification
                if p["description"] == "Auto search":
                    yield ret
                    continue

                # filter
                pp, dp, op = p["(pp, dp, op)"]
                if num_gpus != pp * dp * op:
                    continue

                yield ret