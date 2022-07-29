from typing import Dict, Any
import numpy as np
import alpa

from cluster import ClusterSpec

ParallelSpec = Dict[str, Any]


def precheck(p: ParallelSpec, cluster_spec: ClusterSpec):
    if p["description"] == "Auto search":
        return

    pp, dp, op = p["(pp, dp, op)"]
    physical_mesh_shape = p.get("physical_mesh_shape", (0, 0))
    auto_sharding_option = p.get("auto_sharding_option", {})
    host_ids = p.get("host_ids", None)
    device_ids = p.get("device_ids", None)
    num_auto_layers = p["num_auto_layers"]
    for num_micro_batches in p["num_micro_batches"]:
        _precheck(cluster_spec, pp, dp, op, num_micro_batches, num_auto_layers,
                  host_ids, device_ids)


def _precheck(cluster: ClusterSpec, pp: int, dp: int, op: int,
              num_micro_batches: int, num_auto_layers: int, host_ids,
              device_ids):
    assert num_auto_layers % pp == 0, f"{num_auto_layers} vs {pp}"

    if host_ids is not None:
        assert device_ids is not None
        assert len(host_ids) == pp
        assert len(device_ids) == pp

    assert cluster.num_gpus(
    ) == pp * dp * op, f"{cluster.num_devices} vs {pp} * {dp} * {op}"


def data_parallel(pp: int, dp: int, op: int, num_micro_batches: int,
                  num_auto_layers: int) -> alpa.PipeshardParallel:
    precheck(pp, dp, op, num_micro_batches, num_auto_layers)

    cluster = alpa.get_global_cluster()
    return alpa.PipeshardParallel(
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
