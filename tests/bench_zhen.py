import os

import jax.numpy as jnp
import torch
import numpy as np
import alpa.torch.optim as torchoptim
from alpa.torch.trainer import train_torch_module
import alpa
from alpa.device_mesh import get_global_cluster
from alpa import PipeshardParallel

from torch_frontend.test_zhen import ZHENCollection, TokenMixer


def weight_init_func(pt_module, name_map, params, bufs):
    # for k, m in pt_module.named_modules():
    #     if isinstance(m, torch.nn.Linear):
    #         params[name_map[f"{k}.weight"]] = torch.nn.init.xavier_uniform(params[name_map[f"{k}.weight"]])
    #         params[name_map[f"{k}.bias"]] = torch.nn.init.normal(params[name_map[f"{k}.bias"]], std=1e-6)
    return params, bufs


def train_zhen_homogeneous(input_batch_size, num_iters, parallel_method):
    B = input_batch_size  # 59  # made multiples of 8
    F = 512
    D = 160
    LAYERS = 4
    OUTPUT_PER_ENSEMBLE = 32  # 50  # made multiples of 8
    # OUTPUT_PER_ENSEMBLE = 48  # 50  # made multiples of 8
    # TOKENS = [
    #     TokenMixer.ATTENTION, TokenMixer.LINEAR, TokenMixer.ATTENTION,
    #     TokenMixer.CONVOLUTION, TokenMixer.DOT
    # ]
    TOKENS = [
        TokenMixer.ATTENTION, TokenMixer.LINEAR, TokenMixer.ATTENTION,
        TokenMixer.DOT
    ]

    pt_module_gen = lambda: ZHENCollection(LAYERS, D, TOKENS, F,
                                           OUTPUT_PER_ENSEMBLE)

    # dataloader = [(torch.empty(
    #     B, D, F), torch.empty(B, D * LAYERS * OUTPUT_PER_ENSEMBLE))] * num_iters
    dataloader = [(jnp.ones(
        (B, D, F), jnp.float32), jnp.ones((B, D * LAYERS * OUTPUT_PER_ENSEMBLE), jnp.float32))] * num_iters

    loss_func = lambda *args, **kwargs: torch.nn.functional.mse_loss(
        *args, **kwargs)
    optim_gen = torchoptim.adam(lr=1e-3)


    alpa.global_config.print_compilation_time = True
    alpa.global_config.print_auto_layer_stats = True
    alpa.global_config.xla_client_mem_fraction = 0.7
    alpa.global_config.use_dummy_value_for_benchmarking = True

    train_torch_module(pt_module_gen, weight_init_func, dataloader, loss_func,
                       optim_gen, parallel_method)


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


# {
#     # torchrec: duration of each iteration avg: 0.185473 secs, median: 0.16794457199284807 secs, 90P: 0.16855745000066236 secs, 99P: 1.9236148939817213 secs
#     # num_micro_batches: 8, duration of each iteration avg: 0.693365 secs, median: 0.15849089622497559 secs, 90P: 0.20190882682800293 secs, 99P: 10.839699506759644 secs
#     # num_micro_batches: 2, duration of each iteration avg: 0.693497 secs, median: 0.15735721588134766 secs, 90P: 0.1949770450592041 secs, 99P: 10.804256200790405 secs
#     "description": "data parallel",
#     "(pp, dp, op)": (1, 8, 1),
#     "physical_mesh_shape": (1, 8),
#     "auto_sharding_option": {
#         'force_batch_dim_to_mesh_dim': 0
#     },
# },

# single gpu, duration of each iteration avg: 0.037073 secs, median: 0.032457828521728516 secs, 90P: 0.05207514762878418 secs, 99P: 0.05215716361999512 secs, max_mem_allocated: 13491511040
# 8 gpus, num_micro_batches: 8, duration of each iteration avg: 0.131354 secs, median: 0.13614344596862793 secs, 90P: 0.14940667152404785 secs, 99P: 0.15056109428405762 secs, max_mem_allocated: 3864780032

PARALLEL_METHODS = [
    {
        # num_micro_batches: 32, duration of each iteration avg: 1.116381 secs, median: 1.0280978679656982 secs, 90P: 1.203047752380371 secs, 99P: 3.7230050563812256 secs
        # num_micro_batches: 16, duration of each iteration avg: 0.705956 secs, median: 0.7422900199890137 secs, 90P: 0.8465251922607422 secs, 99P: 1.3067986965179443 secs
        # num_micro_batches: 16, using EFA does not help... duration of each iteration avg: 0.738531 secs, median: 0.7759156227111816 secs, 90P: 0.9488911628723145 secs, 99P: 1.2394416332244873 secs
        # num_micro_batches: 8, duration of each iteration avg: 0.610578 secs, median: 0.6345152854919434 secs, 90P: 0.7277560234069824 secs, 99P: 1.040968894958496 secs

        # num_micro_batches: 16, using EFA, duration of each iteration avg: 0.275258 secs, median: 0.2899587154388428 secs, 90P: 0.34606218338012695 secs, 99P: 0.35033464431762695 secs
        # num_micro_batches: 16, tcp, duration of each iteration avg: 0.356500 secs, median: 0.3932931423187256 secs, 90P: 0.4252331256866455 secs, 99P: 0.4303760528564453 secs
        # This is pytorch ddp: duration of each iteration avg: 0.185125 secs, median: 0.17045736202271655 secs, 90P: 0.17091922700637951 secs, 99P: 1.6407538619823754 secs
        # another run, efa, num_micro_batches: 16, duration of each iteration avg: 0.269500 secs, median: 0.2741248607635498 secs, 90P: 0.3198568820953369 secs, 99P: 0.3624308109283447 secs, max_mem_allocated: 3279045376
        # another run, tcp, num_micro_batches: 16, duration of each iteration avg: 0.238331 secs, median: 0.2433021068572998 secs, 90P: 0.2861208915710449 secs, 99P: 0.31250572204589844 secs, max_mem_allocated: 3279045376
        # efa, 64 gpus, duration of each iteration avg: 0.999151 secs, median: 1.0135297775268555 secs, 90P: 1.223393440246582 secs, 99P: 1.250563383102417 secs, max_mem_allocated: 3027076608
        # tcp, 64 gpus, duration of each iteration avg: 1.143670 secs, median: 1.2758233547210693 secs, 90P: 1.3663244247436523 secs, 99P: 1.367673397064209 secs, max_mem_allocated: 3027076608
        # tcp, 64 gpus, duration of each iteration avg: 1.232506 secs, median: 1.3059947490692139 secs, 90P: 1.3425567150115967 secs, 99P: 1.3427753448486328 secs, max_mem_allocated: 3027076608
        "description": "data parallel",
        "(pp, dp, op)": (1, 8, 1),
        "physical_mesh_shape": (1, 8),
        "auto_sharding_option": {
            'force_batch_dim_to_mesh_dim': 0
        },
    },
    {
        # incompatible shapes before, removing conv2d bypasses this issue
        # num_micro_batches: 32, duration of each iteration avg: 5.999120 secs, median: 6.294488430023193 secs, 90P: 7.15912938117981 secs, 99P: 7.891208648681641 secs
        # num_micro_batches: 16, duration of each iteration avg: 5.624338 secs, median: 5.146093845367432 secs, 90P: 6.106522560119629 secs, 99P: 19.493966579437256 secs
        # num_micro_batches: 8,

        # efa, num_micro_batches: 16, duration of each iteration avg: 0.593172 secs, median: 0.6157050132751465 secs, 90P: 0.6784508228302002 secs, 99P: 0.6992802619934082 secs
        # tcp, num_micro_batches: 16, duration of each iteration avg: 0.669445 secs, median: 0.673079252243042 secs, 90P: 0.8875925540924072 secs, 99P: 0.9069862365722656 secs
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
        # duration of each iteration avg: 1.052618 secs, median: 0.760066032409668 secs, 90P: 1.024703025817871 secs, 99P: 6.421065330505371 secs
        # [2.153844118118286, 2.8738555908203125, 2.2504305839538574]
        # efa, num_micro_batches: 16, duration of each iteration avg: 0.381057 secs, median: 0.3933441638946533 secs, 90P: 0.4248046875 secs, 99P: 0.5040090084075928 secs
        # tcp, num_micro_batches: 16, duration of each iteration avg: 0.410941 secs, median: 0.41552019119262695 secs, 90P: 0.5118517875671387 secs, 99P: 0.5499989986419678 secs
        "description":
            "Example #4: data-parallel across numa-nodes, tensor-parallel within a numa-node",
        "(pp, dp, op)": (1, 4, 4),
        "physical_mesh_shape": (2, 8),
        # "physical_mesh_shape": (0, 0),
        # "host_ids": [[0, 0, 1, 1]],
        # "device_ids": [[[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]]],
        # "host_ids": [[0, 1]],
        # "device_ids": [[[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]],

        # efa, duration of each iteration avg: 1.280525 secs, median: 1.220517635345459 secs, 90P: 1.78116774559021 secs, 99P: 1.839501142501831 secs, max_mem_allocated: 2739700480
        # "(pp, dp, op)": (1, 16, 4),
        # "physical_mesh_shape": (8, 8),
        # efa, duration of each iteration avg: 1.400413 secs, median: 1.4358973503112793 secs, 90P: 1.58518385887146 secs, 99P: 1.7353997230529785 secs, max_mem_allocated: 4391375872
        # "(pp, dp, op)": (1, 8, 8),
        # "physical_mesh_shape": (8, 8),
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
        # num_micro_batches: 32, duration of each iteration avg: 32.164927 secs, median: 31.394895553588867 secs, 90P: 43.208282470703125 secs, 99P: 43.236831188201904 secs
        # num_micro_batches: 32, run again, duration of each iteration avg: 20.166597 secs, median: 21.410356998443604 secs, 90P: 22.74605417251587 secs, 99P: 22.843191146850586 secs
        "description":
            "Example #5.2: data parallel + intra-machine pipeline parallel",
        "(pp, dp, op)": (4, 4, 1),
        # "physical_mesh_shape": (2, 2),  # (4, 1)
        # tcp, num_micro_batches: 16, duration of each iteration avg: 4.716089 secs, median: 4.795248746871948 secs, 90P: 5.876234292984009 secs, 99P: 6.319248199462891 secs
        # [0,3,2,1], [4,7,6,5]
        "host_ids": [[0, 1], [0, 1], [0, 1], [0, 1]],
        "device_ids": [[[0, 4], [0, 4]], [[3, 7], [3, 7]], [[2, 6], [2, 6]],
                       [[1, 5], [1, 5]]],

        # tcp, num_micro_batches: 16, duration of each iteration avg: 4.435792 secs, median: 4.675730228424072 secs, 90P: 4.886426687240601 secs, 99P: 4.9773242473602295 secs
        # [0,3,4,7], [1,2,5,6]
        # "host_ids": [[0, 1], [0, 1], [0, 1], [0, 1]],
        # "device_ids": [[[0, 1], [0, 1]], [[3, 2], [3, 2]], [[4, 5], [4, 5]], [[7, 6], [7, 6]]],

        # duration of each iteration avg: 20.791150 secs, median: 22.128350257873535 secs, 90P: 23.276580095291138 secs, 99P: 23.45837903022766 secs
        # 15.52617883682251, 16.176363706588745, 15.91279649734497, 16.444060564041138, 15.856158018112183, 16.74027109146118, 15.548842191696167, 15.776957750320435
        # tcp, num_micro_batches: 16, duration of each iteration avg: 4.772827 secs, median: 4.985513687133789 secs, 90P: 5.721350908279419 secs, 99P: 5.977432727813721 secs
        # [0,1,4,5], [2,3,6,7]
        # "host_ids": [[0, 1], [0, 1], [0, 1], [0, 1]],
        # "device_ids": [[[0, 2], [0, 2]], [[1, 3], [1, 3]], [[4, 6], [4, 6]], [[5, 7], [5, 7]]],

        # 16.703232765197754, 17.657737255096436, 17.30591320991516, 17.8020601272583, 17.031238794326782, 18.052095651626587, 17.76880693435669, 16.9144070148468
        # another run: 19.64015030860901, 15.822978258132935, 16.193574905395508, 14.976128816604614
        # tcp, num_micro_batches: 16, duration of each iteration avg: 4.822154 secs, median: 5.027798414230347 secs, 90P: 5.656952142715454 secs, 99P: 6.049246072769165 secs
        # [0,1,2,3], [4,5,6,7]
        # "host_ids": [[0, 1], [0, 1], [0, 1], [0, 1]],
        # "device_ids": [[[0, 4], [0, 4]], [[1, 5], [1, 5]], [[2, 6], [2, 6]], [[3, 7], [3, 7]]],
        # wrong
        # tcp, num_micro_batches: 16, duration of each iteration avg: 6.923140 secs, median: 7.217944622039795 secs, 90P: 8.190350532531738 secs, 99P: 8.254141807556152 secs
        # "host_ids": [[0], [1], [0], [1]],
        # "device_ids": [[[0, 1, 2, 3]], [[0, 1, 2, 3]], [[4, 5, 6, 7]], [[4, 5, 6, 7]]],
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


# num_gpus = 16
# avg_batch_size_per_device = 1024
# # input_batch_size = micro_batch_size = avg_batch_size_per_device * num_gpus // dp // num_micro_batches
# [(dp, op)] (pp = 4)
# [16384 / 64 * 16] ->  [(4, 1), (4, 1), (2, 2), (1, 4)]
# [(4, 1), (4, 1), (4, 1), (4, 1)]

BENCHMARK_SUITES = {
    # This is how many samples are processed in one iteration on average to each device.
    # Given the existence of operator-parallel and pipeline-parallel, we may need specify a larger input batch size.
    "avg_batch_size_per_device": 1024,
    "num_micro_batches": 4,
    "num_iters": 20,
    "parallel_methods": PARALLEL_METHODS[0:1],
}


# BENCHMARK_SUITES = {
#     "global_batch_size": 1024 // 2,
#     "num_micro_batches": 8,
#     # actual_batch_per_device = global_batch_size / num_devices
#     "num_iters": 20,
#     "parallel_methods": PARALLEL_METHODS[4:5],
# }


def benchmark_zhen_homogeneous(benchmark_case):
    print(f"benchmarking: {benchmark_case}")
    alpa.init("ray")
    cluster = alpa.device_mesh.get_global_cluster()
    num_gpus = cluster.num_devices

    c = benchmark_case
    avg_batch_size_per_device = c["avg_batch_size_per_device"]
    global_batch_size = avg_batch_size_per_device * num_gpus

    num_micro_batches = c["num_micro_batches"]
    num_iters = c["num_iters"]
    p = c["parallel_method"]

    # high-level parallelism specification
    pp, dp, op = p["(pp, dp, op)"]
    physical_mesh_shape = p.get("physical_mesh_shape", (0, 0))
    auto_sharding_option = p.get("auto_sharding_option", {})
    host_ids = p.get("host_ids", None)
    device_ids = p.get("device_ids", None)

    # input_batch_size = global_batch_size // dp // num_micro_batches
    input_batch_size = global_batch_size
    print(f'avg_batch_size_per_device: {avg_batch_size_per_device}, num_gpus: {num_gpus}, input_batch_size: {input_batch_size}, num_micro_batches: {num_micro_batches}, (pp, dp, op): ({pp}, {dp}, {op})')

    # num_micro_batches = 32
    num_auto_layers = pp * 1

    # check
    precheck(pp, dp, op, num_micro_batches, num_auto_layers, host_ids,
             device_ids)

    physical_mesh_specs = None
    if host_ids is not None:
        physical_mesh_specs = list(zip(host_ids, device_ids))

    # cluster = alpa.device_mesh.get_global_cluster()
    # virtual_mesh = cluster.get_virtual_physical_mesh()
    # logical_mesh_shape = (dp, op)
    # num_mesh_devices = np.prod(logical_mesh_shape)
    # num_devices_per_host = virtual_mesh.num_devices_per_host
    # if num_mesh_devices <= num_devices_per_host:
    #     physical_mesh_shape = (1, num_mesh_devices)
    # else:
    #     assert num_mesh_devices % num_devices_per_host == 0
    #     physical_mesh_shape = (num_mesh_devices // num_devices_per_host,
    #                            num_devices_per_host)

    # data parallel + cross-mesh pipeline parallel + intra-mesh tensor parallel
    parallel_method = PipeshardParallel(
        num_micro_batches=num_micro_batches,
        layer_option=alpa.AutoLayerOption(num_auto_layers),
        stage_option=alpa.ManualStageOption(
            forward_stage_layer_ids=np.array_split(range(num_auto_layers), pp),
            submesh_physical_shapes=[physical_mesh_shape] * pp,
            submesh_logical_shapes=[(dp, op)] * pp,
            submesh_autosharding_option_dicts=[auto_sharding_option] * pp,
            submesh_physical_spec=physical_mesh_specs),
        default_auto_sharding_option=alpa.AutoShardingOption(
            prefer_reduce_scatter=True))
    # {'force_batch_dim_to_mesh_dim': 0}

    # data parallel
    # parallel_method = data_parallel(pp, dp, op, num_micro_batches,
    #                                 num_auto_layers)

    # auto search
    # parallel_method = alpa.PipeshardParallel(
    #     num_micro_batches=num_micro_batches,
    #     layer_option=alpa.AutoLayerOption(layer_num=num_auto_layers),
    #     stage_option="auto")

    # faster auto search
    # parallel_method = PipeshardParallel(
    #     stage_mode="uniform",
    #     num_micro_batches=num_micro_batches,
    #     layer_option=alpa.AutoLayerOption(num_auto_layers, remat_layer=True),
    #     default_auto_sharding_option=alpa.AutoShardingOption(
    #         force_data_parallel=True))

    train_zhen_homogeneous(input_batch_size, num_iters, parallel_method)


def main():
    for parallel_method in BENCHMARK_SUITES["parallel_methods"]:
        case = BENCHMARK_SUITES.copy()
        del case["parallel_methods"]
        case["parallel_method"] = parallel_method
        benchmark_zhen_homogeneous(case)


if __name__ == '__main__':
    # os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    # os.environ["NCCL_DEBUG"] = "info"
    # os.environ["ALPA_USE_AWS_EFA"] = "1"
    main()