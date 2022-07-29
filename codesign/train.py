import os
import time

import torch
import alpa
from alpa.torch.trainer import train_torch_module

from cluster import ClusterSpec, start_ray_cluster
from training_spec import TrainingSpec
from model import create_model, ModelSpec
from db import save_record, save_trial_specs


def weight_init_func(pt_module, name_map, params, bufs):
    return params, bufs


def train_with_alpa(args, cluster_spec: ClusterSpec, model_spec: ModelSpec,
                    training_spec: TrainingSpec,
                    parallel_method: alpa.parallel_method.ParallelMethod):
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
        save_record(cluster_spec, model_spec, training_spec, parallel_method,
                    alpa.global_config, None, None, None, None)
    else:
        save_trial_specs(cluster_spec, model_spec, training_spec,
                         parallel_method)

        try:
            # start ray cluster first
            start_ray_cluster(cluster_spec)
            time.sleep(1)

            # train
            latencies, memory, parallel_plan = train_torch_module(
                pt_module_gen, weight_init_func, dataloader, loss_func,
                optim_gen, parallel_method)

        except Exception as e:
            # catch whatever exception
            print(e)
            save_record(cluster_spec, model_spec, training_spec,
                        parallel_method, alpa.global_config, None, None, None,
                        e)
        else:
            save_record(cluster_spec, model_spec, training_spec,
                        parallel_method, alpa.global_config, latencies, memory,
                        parallel_plan, None)
