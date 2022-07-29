from dataclasses import dataclass
from typing import Dict, Any, Tuple, Sequence, List

from serde import serde
from serde.toml import from_toml

from cluster import ClusterSpec
from training_spec import TrainingSpec
from parallel import ParallelSpec, precheck
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
            tokens = [get_token_mixer(t) for t in c['tokens']]
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

    def generate_parallel_methods(self, cluster_spec: ClusterSpec):
        import alpa
        import numpy as np
        for p in self.parallel_spec:
            for num_micro_batches in p["num_micro_batches"]:
                # high-level parallelism specification
                num_auto_layers = p["num_auto_layers"]
                remat_layer = p["remat_layer"]
                if p["description"] == "Auto search":
                    parallel_method = alpa.PipeshardParallel(
                        num_micro_batches=num_micro_batches,
                        layer_option=alpa.AutoLayerOption(
                            num_auto_layers, remat_layer=remat_layer),
                        stage_option="auto",
                        default_auto_sharding_option=alpa.
                        AutoShardingOption(prefer_reduce_scatter=True))
                    yield parallel_method
                    continue

                # manual options
                pp, dp, op = p["(pp, dp, op)"]
                physical_mesh_shape = p.get("physical_mesh_shape", (0, 0))
                auto_sharding_option = p.get("auto_sharding_option", {})
                host_ids = p.get("host_ids", None)
                device_ids = p.get("device_ids", None)

                # filter
                num_gpus = cluster_spec.num_gpus()
                if num_gpus != pp * dp * op:
                    continue

                physical_mesh_specs = None
                if host_ids is not None:
                    physical_mesh_specs = list(zip(host_ids, device_ids))

                # check
                precheck(p, cluster_spec)

                parallel_method = alpa.PipeshardParallel(
                    num_micro_batches=num_micro_batches,
                    layer_option=alpa.AutoLayerOption(
                        num_auto_layers, remat_layer=remat_layer),
                    stage_option=alpa.ManualStageOption(
                        forward_stage_layer_ids=np.array_split(
                            range(num_auto_layers), pp),
                        submesh_physical_shapes=[physical_mesh_shape] * pp,
                        submesh_logical_shapes=[(dp, op)] * pp,
                        submesh_autosharding_option_dicts=[
                            auto_sharding_option
                        ] * pp,
                        submesh_physical_spec=physical_mesh_specs),
                    default_auto_sharding_option=alpa.AutoShardingOption(
                        prefer_reduce_scatter=True))

                yield parallel_method