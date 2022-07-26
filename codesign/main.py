#!/usr/bin/env python3
import dataclasses
import time
import os
import toml
from typing import Dict, Any, Callable, Sequence

import jax.numpy as jnp
import torch
import numpy as np
import alpa.torch.optim as torchoptim
from alpa.torch.trainer import train_torch_module
import alpa
from alpa.device_mesh import get_global_cluster
from alpa import PipeshardParallel

from zhen import ZHENCollection, TokenMixer

from dataclasses import dataclass
import alpa


def weight_init_func(pt_module, name_map, params, bufs):
    return params, bufs


# The list of environment variable we will print
ENV_FILTER = [
    'NCCL_DEBUG', 'ALPA_USE_AWS_EFA', 'NVIDIA_TF32_OVERRIDE', 'XLA_FLAGS',
    'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'
]

@dataclass
class ClusterSpec(object):
    num_hosts: int
    num_devices_per_host: int
    # ['NVLink', 'EFAx4', 'TCPx4']
    transport: str

    def header_csv(self) -> str:
        return ','.join(['num_nodes', '# gpus_per_node', 'Transport'])

    def value_csv(self) -> str:
        return ','.join([str(v) for v in dataclasses.asdict(self).keys()])


@dataclass
class TrainingSpec(object):
    global_batch_size: int
    avg_batch_size_per_device: int
    num_iters: int
    loss_func: Callable[..., Any]
    optim_gen: Callable[..., Any]

    def header_csv(self) -> str:
        return ','.join(
            ['global batch size', 'avg batch size per gpu', '# iters'])

    def value_csv(self) -> str:
        return ','.join([
            f'{self.global_batch_size}', f'{self.avg_batch_size_per_device}',
            f'{self.num_iters}'
        ])


TRAINING_SPECS = [{
    'avg_batch_size_per_device':
        1024,
    'num_iters':
        20,
    'loss_func':
        lambda *args, **kwargs: torch.nn.functional.mse_loss(*args, **kwargs),
    'optim_gen':
        torchoptim.adam(lr=1e-3),
}]


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


def create_model(model_spec: Dict[str, Any]) -> torch.Module:
    num_features = model_spec['num_features']
    emb_dim = model_spec['emb_dim']
    output_per_emb = model_spec['output_per_emb']
    num_zhen_layers = model_spec['num_zhen_layers']
    tokens = model_spec['tokens']
    return ZHENCollection(num_zhen_layers, emb_dim, tokens, num_features,
                          output_per_emb)


def generate_models(cluster_spec: ClusterSpec):
    # hard coded for now
    configs = open('codesign/models/model.toml', 'r').read()
    for c in configs['model']:
        tokens = [get_token_mixer(t) for t in c['tokens']]
        model_spec = c.copy()
        model_spec['tokens'] = tokens
        model = create_model(model_spec)

        num_hosts = cluster_spec.num_hosts
        num_gpus = num_hosts * cluster_spec.num_devices_per_host
        avg_batch_size_per_device = TRAINING_SPECS[0][
            'avg_batch_size_per_device']
        global_batch_size = avg_batch_size_per_device * num_gpus

        training_spec = TrainingSpec(global_batch_size,
                                     avg_batch_size_per_device,
                                     TRAINING_SPECS[0]['num_iters'],
                                     TRAINING_SPECS[0]['loss_func'],
                                     TRAINING_SPECS[0]['optim_gen'])

        yield model_spec, training_spec, model


def print_trial_specs(cluster_spec: ClusterSpec, model_spec: Dict[str, Any],
                      training_spec: TrainingSpec, parallel_method):
    print(cluster_spec.value_csv())
    print(model_spec)
    print(training_spec.value_csv())
    print(parallel_method)


def print_all_configs(cluster_spec: ClusterSpec, model_spec: Dict[str, Any],
                      training_spec: TrainingSpec, parallel_method,
                      alpa_global_config: alpa.GlobalConfig):
    print(cluster_spec.value_csv())
    print(model_spec)
    print(training_spec.value_csv())
    print(parallel_method)
    print(alpa_global_config)
    print([os.environ[e] for e in ENV_FILTER])

def print_results(latencies: Sequence[float], memory: int, parallel_plan: alpa.parallel_plan.ParallelPlan):
    print(latencies)
    print(memory)
    print(parallel_plan)


def train_with_alpa(args, cluster_spec: ClusterSpec, model_spec, training_spec,
                    model, parallel_method):
    F = model_spec['num_features']
    D = model_spec['emb_dim']
    O = model_spec['output_per_emb']
    L = model_spec['num_zhen_layers']
    B = training_spec['global_batch_size']

    pt_module_gen = lambda: model

    loss_func = training_spec['loss_func']
    optim_gen = training_spec['optim_gen']
    num_iters = training_spec['num_iters']
    dataloader = [torch.empty(B, D, F), torch.empty(B, D * L * O)] * num_iters

    global_batch_size = training_spec.global_batch_size
    avg_batch_size_per_device = training_spec.avg_batch_size_per_device
    num_micro_batches = parallel_method.num_micro_batches
    # Try to make sure global_batch_size // num_micro_batches // mesh[0].dp >= 1024
    # => num_micro_batches * mesh[0].dp <= num_gpus
    if isinstance(parallel_method.stage_option, alpa.ManualStageOption):
        stage0_dp = parallel_method.stage_option.submesh_physical_shapes[0][0]
        assert (
            global_batch_size // num_micro_batches // stage0_dp >=
            avg_batch_size_per_device,
            f"{global_batch_size} // {num_micro_batches} // {stage0_dp} >= {avg_batch_size_per_device}"
        )

    # set environment variables
    transport = training_spec
    if transport == 'NVLink':
        assert cluster_spec.num_hosts == 1, f"{cluster_spec}"
    elif transport == 'EFAv4':
        assert cluster_spec.num_hosts > 1, f"{cluster_spec}"
        os.environ['ALPA_USE_AWS_EFA'] = '1'
    elif transport == 'TCPv4':
        os.environ['ALPA_USE_AWS_EFA'] = '0'
    else:
        raise NotImplementedError(f'{transport}')

    # alpa global config to customize
    alpa.global_config.print_compilation_time = True
    alpa.global_config.print_auto_layer_stats = True
    alpa.global_config.xla_client_mem_fraction = 0.7
    alpa.global_config.use_dummy_value_for_benchmarking = True

    # TODO(cjr): store them to databases, add datetime for each record
    # TODO(cjr): FpP, activation size, model size, jaxpr
    print_trial_specs(cluster_spec, model_spec, training_spec, parallel_method)
    if args.dry_run:
        latencies, memory, parallel_plan = train_torch_module(
            pt_module_gen, weight_init_func, dataloader, loss_func, optim_gen,
            parallel_method)
        print_results(latencies, memory, parallel_plan)
    print_all_configs(cluster_spec, model_spec, training_spec, parallel_method,
                      alpa.global_config)


def precheck(pp: int, dp: int, op: int, num_micro_batches: int,
             num_auto_layers: int, host_ids, device_ids):
    assert num_auto_layers % pp == 0, f"{num_auto_layers} vs {pp}"

    if host_ids is not None:
        assert device_ids is not None
        assert len(host_ids) == pp
        assert len(device_ids) == pp

    alpa.init("ray")
    cluster = alpa.device_mesh.get_global_cluster()
    assert cluster.num_devices == pp * dp * op, f"{cluster.num_devices} vs {pp} * {dp} * {op}"


def data_parallel(pp: int, dp: int, op: int, num_micro_batches: int,
                  num_auto_layers: int) -> PipeshardParallel:
    precheck(pp, dp, op, num_micro_batches, num_auto_layers)

    cluster = get_global_cluster()
    return PipeshardParallel(
        num_micro_batches=num_micro_batches,
        layer_option=alpa.AutoLayerOption(num_auto_layers),
        stage_option=alpa.ManualStageOption(
            forward_stage_layer_ids=np.array_split(range(num_auto_layers), pp),
            submesh_physical_shapes=[cluster.get_virtual_physical_mesh().shape
                                    ] * pp,
            submesh_logical_shapes=[(dp, op)] * pp,
            submesh_autosharding_option_dicts=[{
                'force_batch_dim_to_mesh_dim': 0
            }] * pp),
        default_auto_sharding_option=alpa.AutoShardingOption(
            prefer_reduce_scatter=True))


CLUSTER_SPECS = [
    ClusterSpec(1, 1, 'NVLink'),
    ClusterSpec(1, 8, 'NVLink'),
    ClusterSpec(2, 8, 'EFAx4'),
    ClusterSpec(2, 8, 'TCPx4'),
    ClusterSpec(8, 8, 'EFAx4'),
    ClusterSpec(8, 8, 'TCPx4'),
]

PARALLEL_METHODS = [
    {
        "description": "data parallel",
        "num_auto_layers": 1,
        "remat_layer": False,
        "num_micro_batches": [1],  # explode this
        "(pp, dp, op)": (1, 8, 1),
        "physical_mesh_shape": (1, 8),
        "auto_sharding_option": {
            'force_batch_dim_to_mesh_dim': 0
        },
    },
    {
        # incompatible shapes before, removing conv2d works around this issue
        "description": "two nodes, within each node running operator parallel",
        "(pp, dp, op)": (1, 2, 8),
        "physical_mesh_shape": (2, 8),
    },
    {
        # this plan looks suboptimal, low priority
        # failed: jax._src.traceback_util.UnfilteredStackTrace: TypeError: lax.dynamic_update_slice requires arguments to have the same dtypes, got bool, uint8.
        "description":
            "Example #1: two data pipelines (dp=2), each data pipeline has 4 pipeline stages (pp=4), within each pipeline stages, run tensor parallel on two GPUs.",
        "(pp, dp, op)": (4, 2, 2),
        "physical_mesh_shape": (1, 4),
    },
    {
        # failed: never finish the first iteration
        "description":
            "Example #2: intra-numa-node (4 gpus) operator parallel, cross-numa and cross-machine pipeline",
        "(pp, dp, op)": (4, 1, 4),
        "physical_mesh_shape": (1, 4),
    },
    {
        # This plan looks not bad
        "description":
            "Example #4: data-parallel across numa-nodes, tensor-parallel within a numa-node",
        "(pp, dp, op)": (1, 4, 4),
        "physical_mesh_shape": (2, 8),
    },
    {
        # failed: never finish the first iteration
        "description":
            "Example #5.1: data parallel + cross-machine pipeline parallel",
        "(pp, dp, op)": (4, 4, 1),
        "physical_mesh_shape": (1, 4),
    },
    {
        # This plan looks not bad
        "description":
            "Example #5.2: data parallel + intra-machine pipeline parallel",
        "(pp, dp, op)": (4, 4, 1),
        "host_ids": [[0, 1], [0, 1], [0, 1], [0, 1]],
        "device_ids": [[[0, 4], [0, 4]], [[3, 7], [3, 7]], [[2, 6], [2, 6]],
                       [[1, 5], [1, 5]]],
    },
    {
        # this looks suboptimal
        # failed: assert tuple(expected_microbatched_shape) == microbatch_shape, AssertionError: (1, 128) vs (128,)
        "description":
            "two gpus a group, each group in charge of a pipeline stage (8 stages)",
        "(pp, dp, op)": (8, 1, 2),
        "physical_mesh_shape": (1, 2),
    },
]


def generate_parallel_methods():
    for p in PARALLEL_METHODS["parallel_methods"]:
        num_micro_batches = p["num_micro_batches"]

        # high-level parallelism specification
        pp, dp, op = p["(pp, dp, op)"]
        physical_mesh_shape = p.get("physical_mesh_shape", (0, 0))
        auto_sharding_option = p.get("auto_sharding_option", {})
        host_ids = p.get("host_ids", None)
        device_ids = p.get("device_ids", None)
        num_auto_layers = p["num_auto_layers"]

        # check
        precheck(pp, dp, op, num_micro_batches, num_auto_layers, host_ids,
                 device_ids)

        physical_mesh_specs = None
        if host_ids is not None:
            physical_mesh_specs = list(zip(host_ids, device_ids))

        parallel_method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            layer_option=alpa.AutoLayerOption(num_auto_layers),
            stage_option=alpa.ManualStageOption(
                forward_stage_layer_ids=np.array_split(range(num_auto_layers),
                                                       pp),
                submesh_physical_shapes=[physical_mesh_shape] * pp,
                submesh_logical_shapes=[(dp, op)] * pp,
                submesh_autosharding_option_dicts=[auto_sharding_option] * pp,
                submesh_physical_spec=physical_mesh_specs),
            default_auto_sharding_option=alpa.AutoShardingOption(
                prefer_reduce_scatter=True))

        yield parallel_method


def shutdown_ray_cluster():
    pass


def start_ray_cluster(cluster_spec: ClusterSpec):
    shutdown_ray_cluster()
    pass


'''
example: ./main.py --dry-run
example: NCCL_DEBUG=info ./main.py
'''

import argparse


def add_args(parser):
    parser.add_argument(
        '-D',
        '--dry-run',
        type=bool,
        action='store_true',
        help='Do not run. Only print what settings was chosen to run.')


def main(args):
    for cluster_spec in CLUSTER_SPECS:
        start_ray_cluster(cluster_spec)
        for model_spec, training_spec, model in generate_models(cluster_spec):
            for parallel_method in generate_parallel_methods():
                time.sleep(1)
                train_with_alpa(args, cluster_spec, model_spec, training_spec,
                                model, parallel_method)


if __name__ == '__main__':
    # os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    # os.environ["NCCL_DEBUG"] = "info"
    # os.environ["ALPA_USE_AWS_EFA"] = "1"
    # parse args
    parser = argparse.ArgumentParser(
        description="allreduce completion time calculator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_args(parser)
    args = parser.parse_args()
    main(args)