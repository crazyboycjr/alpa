#!/usr/bin/env python3
import dataclasses
import time
import os
import toml
from dataclasses import dataclass
from typing import Dict, Any, Callable, Sequence, Optional, Tuple

import ray
import torch
import numpy as np
import alpa.torch.optim as torchoptim
from alpa.torch.trainer import train_torch_module
import alpa
from alpa.device_mesh import get_global_cluster
from alpa import PipeshardParallel

from zhen import ZHENCollection, TokenMixer


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

    def num_gpus(self) -> int:
        return self.num_hosts * self.num_devices_per_host

    def header_csv(self) -> str:
        return ','.join(['num_nodes', '# gpus_per_node', 'Transport'])

    def value_csv(self) -> str:
        return ','.join([str(v) for v in dataclasses.asdict(self).values()])


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


def create_model(model_spec: Dict[str, Any]):
    num_features = model_spec['num_features']
    emb_dim = model_spec['emb_dim']
    output_per_emb = model_spec['output_per_emb']
    num_zhen_layers = model_spec['num_zhen_layers']
    tokens = model_spec['tokens']
    return ZHENCollection(num_zhen_layers, emb_dim, tokens, num_features,
                          output_per_emb)


def generate_models(
        cluster_spec: ClusterSpec) -> Tuple[Dict[str, Any], TrainingSpec]:
    # hard coded for now
    content = open('codesign/models/model.toml', 'r').read()
    configs = toml.loads(content)
    for c in configs['model']:
        tokens = [get_token_mixer(t) for t in c['tokens']]
        model_spec = c.copy()
        model_spec['tokens'] = tokens

        num_gpus = cluster_spec.num_gpus()
        avg_batch_size_per_device = TRAINING_SPECS[0][
            'avg_batch_size_per_device']
        global_batch_size = avg_batch_size_per_device * num_gpus

        training_spec = TrainingSpec(global_batch_size,
                                     avg_batch_size_per_device,
                                     TRAINING_SPECS[0]['num_iters'],
                                     TRAINING_SPECS[0]['loss_func'],
                                     TRAINING_SPECS[0]['optim_gen'])

        yield model_spec, training_spec


def print_trial_specs(cluster_spec: ClusterSpec, model_spec: Dict[str, Any],
                      training_spec: TrainingSpec, parallel_method):
    print(cluster_spec.value_csv())
    print(model_spec)
    print(training_spec.value_csv())
    print(parallel_method)


def print_all(cluster_spec: ClusterSpec, model_spec: Dict[str, Any],
              training_spec: TrainingSpec, parallel_method,
              alpa_global_config: alpa.global_env.GlobalConfig,
              latencies: Optional[Sequence[float]], memory: Optional[int],
              parallel_plan: Optional[alpa.parallel_plan.ParallelPlan],
              error: Optional[Exception]):
    print(cluster_spec.value_csv())
    print(model_spec)
    print(training_spec.value_csv())
    print(parallel_method)
    print(alpa_global_config)
    print([f'{e}={os.getenv(e)}' for e in ENV_FILTER if e in os.environ])
    if latencies is not None:
        print(latencies)
    if memory is not None:
        print(memory)
    if parallel_plan is not None:
        print(parallel_plan)
    if error is not None:
        print(parallel_plan)


def print_results(latencies: Sequence[float], memory: int,
                  parallel_plan: alpa.parallel_plan.ParallelPlan):
    print(latencies)
    print(memory)
    print(parallel_plan)


def train_with_alpa(args, cluster_spec: ClusterSpec, model_spec,
                    training_spec: TrainingSpec, parallel_method):
    F = model_spec['num_features']
    D = model_spec['emb_dim']
    O = model_spec['output_per_emb']
    L = model_spec['num_zhen_layers']
    B = training_spec.global_batch_size

    # This style force the model to be created early, which compromise the purpose of deferred initialization
    # pt_module_gen = lambda: model
    pt_module_gen = lambda: create_model(model_spec)

    loss_func = training_spec.loss_func
    optim_gen = training_spec.optim_gen
    num_iters = training_spec.num_iters
    dataloader = [(torch.empty(B, D, F), torch.empty(B, D * L * O))] * num_iters

    global_batch_size = training_spec.global_batch_size
    avg_batch_size_per_device = training_spec.avg_batch_size_per_device
    num_micro_batches = parallel_method.num_micro_batches
    # Try to make sure global_batch_size // num_micro_batches // mesh[0].dp >= 1024
    # => num_micro_batches <= num_gpus // mesh[0].dp
    if isinstance(
            parallel_method.stage_option, alpa.ManualStageOption
    ) and parallel_method.stage_option.submesh_physical_spec is None:
        stage0_dp = parallel_method.stage_option.submesh_logical_shapes[0][0]
        assert global_batch_size // num_micro_batches // stage0_dp >= avg_batch_size_per_device, "{} // {} // {} >= {}".format(
            global_batch_size, num_micro_batches, stage0_dp,
            avg_batch_size_per_device)

    # set environment variables
    transport = cluster_spec.transport
    if transport == 'NVLink':
        assert cluster_spec.num_hosts == 1, f"{cluster_spec}"
    elif transport == 'EFAx4':
        assert cluster_spec.num_hosts > 1, f"{cluster_spec}"
        os.environ['ALPA_USE_AWS_EFA'] = '1'
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/efa/lib'
        alpa.global_config.use_aws_efa = True
    elif transport == 'TCPx4':
        alpa.global_config.use_aws_efa = False
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
    if args.dry_run:
        print_all(cluster_spec, model_spec, training_spec, parallel_method,
                  alpa.global_config, None, None, None, None)
    else:
        print_trial_specs(cluster_spec, model_spec, training_spec,
                          parallel_method)

        try:
            latencies, memory, parallel_plan = train_torch_module(
                pt_module_gen, weight_init_func, dataloader, loss_func,
                optim_gen, parallel_method)
        except Exception as e:
            # catch whatever exception
            print(e)
            print_all(cluster_spec, model_spec, training_spec, parallel_method,
                      alpa.global_config, None, None, None, e)
        else:
            print_all(cluster_spec, model_spec, training_spec, parallel_method,
                      alpa.global_config, latencies, memory, parallel_plan,
                      None)


def precheck(pp: int, dp: int, op: int, num_micro_batches: int,
             num_auto_layers: int, host_ids, device_ids):
    assert num_auto_layers % pp == 0, f"{num_auto_layers} vs {pp}"

    if host_ids is not None:
        assert device_ids is not None
        assert len(host_ids) == pp
        assert len(device_ids) == pp

    alpa.init("ray")
    cluster = alpa.device_mesh.get_global_cluster()
    alpa.shutdown()
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
    # ClusterSpec(2, 8, 'TCPx4'),
    ClusterSpec(8, 8, 'EFAx4'),
    # ClusterSpec(8, 8, 'TCPx4'),
]

PARALLEL_METHODS = [
    {
        "description": "Auto search",
        "num_micro_batches": [1, 2, 4],  # explode this
        "num_auto_layers": 4,
        "remat_layer": False,
    },
    {
        "description": "data parallel",
        "num_micro_batches": [1],  # explode this
        "(pp, dp, op)": (1, 16, 1),
        "physical_mesh_shape": (2, 8),
        "auto_sharding_option": {
            'force_batch_dim_to_mesh_dim': 0
        },
        "num_auto_layers": 1,
        "remat_layer": False,
    },
    {
        "description": "data parallel",
        "num_micro_batches": [1],  # explode this
        "(pp, dp, op)": (1, 64, 1),
        "physical_mesh_shape": (8, 8),
        "auto_sharding_option": {
            'force_batch_dim_to_mesh_dim': 0
        },
        "num_auto_layers": 1,
        "remat_layer": False,
    },
    {
        # incompatible shapes before, removing conv2d works around this issue
        "description": "two nodes, within each node running operator parallel",
        "num_micro_batches": [1, 2, 4, 8],
        "(pp, dp, op)": (1, 2, 8),
        "physical_mesh_shape": (2, 8),
        "num_auto_layers": 1,
        "remat_layer": False,
    },
    {
        # this plan looks suboptimal, low priority
        # failed: jax._src.traceback_util.UnfilteredStackTrace: TypeError: lax.dynamic_update_slice requires arguments to have the same dtypes, got bool, uint8.
        "description":
            "Example #1: two data pipelines (dp=2), each data pipeline has 4 pipeline stages (pp=4), within each pipeline stages, run tensor parallel on two GPUs.",
        "num_micro_batches": [1, 2, 4, 8],
        "(pp, dp, op)": (4, 2, 2),
        "physical_mesh_shape": (1, 4),
        "num_auto_layers":
            4,
        "remat_layer":
            False,
    },
    {
        # failed: never finish the first iteration
        "description":
            "Example #2: intra-numa-node (4 gpus) operator parallel, cross-numa and cross-machine pipeline",
        "num_micro_batches": [1, 2, 4, 8, 16],
        "(pp, dp, op)": (4, 1, 4),
        "physical_mesh_shape": (1, 4),
        "num_auto_layers":
            4,
        "remat_layer":
            False,
    },
    {
        # This plan looks not bad
        "description":
            "Example #4: data-parallel across numa-nodes, tensor-parallel within a numa-node",
        "num_micro_batches": [1, 2, 4],
        "(pp, dp, op)": (1, 4, 4),
        "physical_mesh_shape": (2, 8),
        "num_auto_layers":
            1,
        "remat_layer":
            False,
    },
    {
        # failed: never finish the first iteration
        "description":
            "Example #5.1: data parallel + cross-machine pipeline parallel",
        "num_micro_batches": [1, 2, 4],
        "(pp, dp, op)": (4, 4, 1),
        "physical_mesh_shape": (1, 4),
        "num_auto_layers":
            4,
        "remat_layer":
            False,
    },
    {
        # This plan looks not bad
        "description":
            "Example #5.2: data parallel + intra-machine pipeline parallel",
        "num_micro_batches": [1, 2, 4],
        "(pp, dp, op)": (4, 4, 1),
        "host_ids": [[0, 1], [0, 1], [0, 1], [0, 1]],
        "device_ids": [[[0, 4], [0, 4]], [[3, 7], [3, 7]], [[2, 6], [2, 6]],
                       [[1, 5], [1, 5]]],
        "num_auto_layers":
            4,
        "remat_layer":
            False,
    },
    {
        # this looks suboptimal
        # failed: assert tuple(expected_microbatched_shape) == microbatch_shape, AssertionError: (1, 128) vs (128,)
        "description":
            "two gpus a group, each group in charge of a pipeline stage (8 stages)",
        "num_micro_batches": [1, 2, 4, 8, 16],
        "(pp, dp, op)": (8, 1, 2),
        "physical_mesh_shape": (1, 2),
        "num_auto_layers":
            8,
        "remat_layer":
            False,
    },
]


def generate_parallel_methods(cluster_spec: ClusterSpec):
    # for p in PARALLEL_METHODS:
    for p in PARALLEL_METHODS[3:4]:
        for num_micro_batches in p["num_micro_batches"]:
            # high-level parallelism specification
            num_auto_layers = p["num_auto_layers"]
            remat_layer = p["remat_layer"]
            if p["description"] == "Auto search":
                parallel_method = PipeshardParallel(
                    num_micro_batches=num_micro_batches,
                    layer_option=alpa.AutoLayerOption(num_auto_layers,
                                                      remat_layer=remat_layer),
                    stage_option="auto",
                    default_auto_sharding_option=alpa.AutoShardingOption(
                        prefer_reduce_scatter=True))
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

            # check
            precheck(pp, dp, op, num_micro_batches, num_auto_layers, host_ids,
                     device_ids)

            physical_mesh_specs = None
            if host_ids is not None:
                physical_mesh_specs = list(zip(host_ids, device_ids))

            parallel_method = PipeshardParallel(
                num_micro_batches=num_micro_batches,
                layer_option=alpa.AutoLayerOption(num_auto_layers,
                                                  remat_layer=remat_layer),
                stage_option=alpa.ManualStageOption(
                    forward_stage_layer_ids=np.array_split(
                        range(num_auto_layers), pp),
                    submesh_physical_shapes=[physical_mesh_shape] * pp,
                    submesh_logical_shapes=[(dp, op)] * pp,
                    submesh_autosharding_option_dicts=[auto_sharding_option] *
                    pp,
                    submesh_physical_spec=physical_mesh_specs),
                default_auto_sharding_option=alpa.AutoShardingOption(
                    prefer_reduce_scatter=True))

            yield parallel_method


import threading
import subprocess
import shlex
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=
    '%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def run_task(cmd):
    #cmd_snip = shlex.split(cmd + " i am " + str(tid))
    cmd_snip = shlex.split(cmd)
    p = subprocess.Popen(cmd_snip,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()
    #print 'out:', out
    #print 'err:', err
    #print 'rc:', p.returncode
    return out, err


# Create threads. Each threads is responsible to start all the commands on a host.
def ssh_submit(cmds, ths):
    for k, v in cmds.items():
        cmd_on_host = ';'.join(v + ['wait'])
        cmd = 'ssh -p 22 -o StrictHostKeyChecking=no {} "{}"'.format(
            k, cmd_on_host)
        print(cmd)
        ths.append(threading.Thread(target=run_task, args=(cmd,)))


HEAD = "a8"
FOLLOWERS = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
RAY_PATH = "/home/cjr/miniconda3/envs/alpa/bin/ray"


def wait_for_threads(ths):
    logger.info("starting threads")
    for th in ths:
        th.start()
    logger.info("waiting for threads to finish")
    for th in ths:
        th.join()
    logger.info("threads finished")


# shutdown ray on all machines
def shutdown_ray_cluster():
    ray.shutdown()

    th_leader = []
    ths_follower = []
    cmd_leader = {
        HEAD: [f"{RAY_PATH} stop --grace-period 10"],
    }
    cmd_follower = {}
    for w in FOLLOWERS:
        cmd_follower[w] = [f"{RAY_PATH} stop --grace-period 10"]

    ssh_submit(cmd_follower, ths_follower)
    wait_for_threads(ths_follower)

    ssh_submit(cmd_leader, th_leader)
    wait_for_threads(th_leader)


def start_ray_cluster(cluster_spec: ClusterSpec):
    assert cluster_spec.num_hosts <= len(
        FOLLOWERS) + 1 and cluster_spec.num_hosts > 0
    # shutdown cluster first
    shutdown_ray_cluster()

    time.sleep(2)
    th_leader = []
    ths_follower = []
    th_checker = []
    cmd_leader = {
        HEAD: [
            f"{RAY_PATH} start --head --dashboard-host 0.0.0.0 --num-gpus {cluster_spec.num_devices_per_host}"
        ],
    }
    cmd_follower = {}
    cmd_checker = {HEAD: [f"{RAY_PATH} status", f"{RAY_PATH} status"]}
    for i in range(1, cluster_spec.num_hosts):
        cmd_follower[FOLLOWERS[i - 1]] = [
            f"{RAY_PATH} start --num-gpus {cluster_spec.num_devices_per_host} --address=a8:6379"
        ]
    ssh_submit(cmd_leader, th_leader)
    wait_for_threads(th_leader)

    ssh_submit(cmd_follower, ths_follower)
    wait_for_threads(ths_follower)

    ssh_submit(cmd_checker, th_checker)
    wait_for_threads(th_checker)
    ray.init(address="auto", ignore_reinit_error=True)


'''
example: ./main.py --dry-run
example: NCCL_DEBUG=info ./main.py
'''

import argparse


def add_args(parser):
    parser.add_argument(
        '-D',
        '--dry-run',
        action='store_true',
        help='Do not run. Only print what settings was chosen to run.')


def main(args):
    for cluster_spec in CLUSTER_SPECS:
        start_ray_cluster(cluster_spec)
        for model_spec, training_spec in generate_models(cluster_spec):
            for parallel_method in generate_parallel_methods(cluster_spec):
                time.sleep(1)
                train_with_alpa(args, cluster_spec, model_spec, training_spec,
                                parallel_method)


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