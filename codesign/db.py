import time
import logging
import os
import dataclasses
from typing import Dict, Optional, Any, Sequence
import sqlite3
import uuid
import numpy as np

import alpa
from alpa.parallel_plan import ParallelPlan
from alpa.global_env import GlobalConfig

import model
import parallel
from cluster import ClusterSpec
from parallel import ParallelSpec
from training_spec import TrainingSpec
from log import logger


# The list of environment variable we will print
ENV_FILTER = [
    'NCCL_DEBUG', 'ALPA_USE_AWS_EFA', 'NVIDIA_TF32_OVERRIDE', 'XLA_FLAGS',
    'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'
]

# num_nodes	gpus_per_node	Transport	input_batch_size	avg_batch_size_per_gpu	parallel strategy desc	num_micro_batches	num_auto_layers	(pp, dp, op)	physical_mesh_shapes	host_ids	device_ids	re-materialization	avg (s)	median (s)	p90 (s)	Max Memory (Bytes)


def print_trial_specs(cluster_spec: ClusterSpec, model_spec: Dict[str, Any],
                      training_spec: TrainingSpec, parallel_spec: ParallelSpec):
    print(cluster_spec)
    print(model_spec)
    print(training_spec)
    print(parallel_spec)


def print_results(latencies: Sequence[float], memory: int,
                  parallel_plan: alpa.parallel_plan.ParallelPlan):
    print(latencies)
    print(memory)
    print(parallel_plan)


def get_all_configs(parallel_method: alpa.parallel_method.ParallelMethod,
                    parallel_plan: Optional[alpa.parallel_plan.ParallelPlan],
                    alpa_global_config: GlobalConfig) -> str:
    import sys
    import copy
    pm = copy.deepcopy(parallel_method)
    pp = copy.deepcopy(parallel_plan)

    # manually expand parallel_method
    pm_dict = pm.__dict__
    # if pm_dict.get("devices", None) is not None:
    #     pm_dict["devices"] = pm_dict["devices"].__dict__
    # if pm_dict.get("as_option", None) is not None and not isinstance(
    #         pm_dict.get("as_option", None), dict):
    #     pm_dict["as_option"] = pm_dict["as_option"].__dict__
    if not isinstance(pm_dict.get("layer_option", ""), str) and not isinstance(
            pm_dict.get("layer_option", ""), dict):
        pm_dict["layer_option"] = pm_dict["layer_option"].__dict__
    # if not isinstance(pm_dict.get("stage_option", ""), str) and not isinstance(
    #         pm_dict.get("stage_option", ""), dict):
    #     pm_dict["stage_option"] = pm_dict["stage_option"].__dict__
    # print(pm_dict)
    # sys.exit(1)

    if pp is not None:
        if not isinstance(pp.pipeline_plan.layer_option,
                          str) and not isinstance(
                              pp.pipeline_plan.layer_option, dict):
            pp.pipeline_plan.layer_option = pp.pipeline_plan.layer_option.__dict__
        plan = str(dataclasses.asdict(parallel_plan))
    else:
        plan = None
    all_configs = {
        'parallel_method': pm_dict,
        'parallel_plan': plan,
        'alpa_global_config': alpa_global_config.__dict__,
        'env': [f'{e}={os.getenv(e)}' for e in ENV_FILTER if e in os.environ],
    }
    # return json.dumps(all_configs)
    return str(all_configs)


def print_record(uuid: uuid.UUID, cluster_spec: ClusterSpec,
                 model_spec: Dict[str, Any], training_spec: TrainingSpec,
                 parallel_spec: ParallelSpec,
                 parallel_method: alpa.parallel_method.ParallelMethod,
                 alpa_global_config: GlobalConfig,
                 latencies: Optional[Sequence[float]], memory: Optional[int],
                 parallel_plan: Optional[ParallelPlan],
                 error: Optional[Exception]):
    print('job_id:', uuid)
    print(cluster_spec)
    print(model_spec)
    print(training_spec)
    print(parallel_spec)
    # print(parallel_method.__dict__)
    # print(alpa_global_config.__dict__)
    # print([f'{e}={os.getenv(e)}' for e in ENV_FILTER if e in os.environ])
    print(get_all_configs(parallel_method, parallel_plan, alpa_global_config))
    if latencies is not None:
        print(latencies)
    if memory is not None:
        print(memory)
    if parallel_plan is not None:
        print(parallel_plan)
    if error is not None:
        print(error)


class DB(object):

    def __init__(self, db_file: str):
        '''Connect and create the result database.'''
        self.con = sqlite3.connect(db_file)
        self.cur = self.con.cursor()

        # this is the only table we use
        self.create_table()

    def create_table(self):
        # Create table
        self.cur.execute('''CREATE TABLE IF NOT EXISTS results
                            (datetime text, job_duration real, trail_uuid text,
                            num_zhen_layers int, token_mixer text, input_features int, emb_dimensions int, output_per_ensemble int,
                            num_nodes int, gpus_per_node int, transport text,
                            input_batch_size int, avg_batch_size_per_gpu int, num_iters int,
                            description text, num_micro_batches int, num_auto_layers int, pp_dp_op text, physical_mesh_shapes text, host_ids text, device_ids text, remat_layer int,
                            avg real, median real, p90 real, max_memory int,
                            error text,
                            all_configs text)''')
        # job duration is for tracking the progress, just providing an estimation

    def __del__(self):
        # Save (commit) the changes
        self.con.commit()

        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        self.con.close()

    def query_record(self, cluster_spec: ClusterSpec,
                     model_spec: Dict[str, Any], training_spec: TrainingSpec,
                     parallel_spec: ParallelSpec,
                     retry_failed: bool = False) -> bool:
        '''Query if the record exists. True for found; False for not found.'''
        model_spec_sql_val = model.to_sql_values(model_spec)
        cluster_spec_sql_val = cluster_spec.to_sql_values()
        training_spec_sql_val = training_spec.to_sql_values()
        parallel_spec_sql_val = parallel.to_sql_values(parallel_spec)

        # select rowid from results where (num_zhen_layers, input_features, input_batch_size) = (4, 512, 8193);
        sql_cmd = f'''SELECT rowid FROM results where
                      (
                        num_zhen_layers, token_mixer, input_features, emb_dimensions, output_per_ensemble,
                        num_nodes, gpus_per_node, transport,
                        input_batch_size, avg_batch_size_per_gpu, num_iters,
                        description, num_micro_batches, num_auto_layers, pp_dp_op, physical_mesh_shapes, host_ids, device_ids, remat_layer
                      ) IS (
                        {model_spec_sql_val},
                        {cluster_spec_sql_val},
                        {training_spec_sql_val},
                        {parallel_spec_sql_val}
                      )'''
        if retry_failed:
            sql_cmd += ' and avg IS NOT NULL'

        logger.debug(sql_cmd)
        return self.cur.execute(sql_cmd).fetchone() is not None

    def save_record(self, uuid: uuid.UUID, cluster_spec: ClusterSpec,
                    model_spec: Dict[str, Any], training_spec: TrainingSpec,
                    parallel_spec: ParallelSpec,
                    parallel_method: alpa.parallel_method.ParallelMethod,
                    alpa_global_config: GlobalConfig,
                    latencies: Optional[Sequence[float]], memory: Optional[int],
                    parallel_plan: Optional[ParallelPlan],
                    error: Optional[Exception], start_ts):
        job_duration = time.perf_counter() - start_ts

        print_record(uuid, cluster_spec, model_spec, training_spec,
                     parallel_spec, parallel_method, alpa.global_config,
                     alpa_global_config, latencies, parallel_plan, error)

        model_spec_sql_val = model.to_sql_values(model_spec)
        cluster_spec_sql_val = cluster_spec.to_sql_values()
        training_spec_sql_val = training_spec.to_sql_values()
        parallel_spec_sql_val = parallel.to_sql_values(parallel_spec)
        if latencies is not None:
            avg_lat = np.mean(latencies)
            med_lat = np.median(latencies)
            p90_lat = np.percentile(latencies, 90)
        else:
            avg_lat = med_lat = p90_lat = 'NULL'
        max_memory = memory if memory is not None else 'NULL'
        error = f'"{str(error)}"' if error is not None else 'NULL'

        all_configs = get_all_configs(parallel_method, parallel_plan,
                                      alpa_global_config)
        sql_cmd = f'''INSERT INTO results VALUES
                      (
                        datetime(), {job_duration}, "{uuid}",
                        {model_spec_sql_val},
                        {cluster_spec_sql_val},
                        {training_spec_sql_val},
                        {parallel_spec_sql_val},
                        {avg_lat}, {med_lat}, {p90_lat}, {max_memory},
                        {error},
                        ?
                      )'''

        logger.debug(sql_cmd)
        self.cur.execute(sql_cmd, (all_configs,))

        # Save (commit) the changes
        self.con.commit()