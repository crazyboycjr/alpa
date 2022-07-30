import os
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
                    parallel_plan: alpa.parallel_plan.ParallelPlan,
                    alpa_global_config: GlobalConfig) -> str:
    import json
    all_configs = {
        'parallel_method': parallel_method.__dict__,
        'parallel_plan': parallel_plan,
        'alpa_global_config': alpa_global_config.__dict__,
        'env': [f'{e}={os.getenv(e)}' for e in ENV_FILTER if e in os.environ],
    }
    return json.dumps(all_configs)


def print_record(cluster_spec: ClusterSpec, model_spec: Dict[str, Any],
                 training_spec: TrainingSpec, parallel_spec: ParallelSpec,
                 parallel_method: alpa.parallel_method.ParallelMethod,
                 alpa_global_config: GlobalConfig,
                 latencies: Optional[Sequence[float]], memory: Optional[int],
                 parallel_plan: Optional[ParallelPlan],
                 error: Optional[Exception]):
    print(cluster_spec.value_csv())
    print(model_spec)
    print(training_spec.value_csv())
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
        self.con = sqlite3.connect('results.db')
        self.cur = self.con.cursor()

        # this is the only table we use
        self.create_table()

    def create_table(self):
        # Create table
        self.cur.execute('''CREATE TABLE IF NOT EXISTS results
                            (datetime text, trail_uuid text,
                            num_zhen_layers int, token_mixer text, input_features int, emb_dimensions int, output_per_ensemble int
                            num_nodes int, gpus_per_node int, transport text,
                            input_batch_size int, avg_batch_size_per_gpu int, num_iters int,
                            description text, num_micro_batches int, num_auto_layers int, pp_dp_op text, physical_mesh_shapes text, host_ids text, device_ids text, remat_layer int,
                            avg real, median real, p90 real, max_memory int,
                            all_configs, text)''')

    def __del__(self):
        # Save (commit) the changes
        self.con.commit()

        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        self.con.close()

    def save_record(self, uuid: uuid.UUID, cluster_spec: ClusterSpec,
                    model_spec: Dict[str, Any], training_spec: TrainingSpec,
                    parallel_spec: ParallelSpec,
                    parallel_method: alpa.parallel_method.ParallelMethod,
                    alpa_global_config: GlobalConfig,
                    latencies: Optional[Sequence[float]], memory: Optional[int],
                    parallel_plan: Optional[ParallelPlan],
                    error: Optional[Exception]):
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
        error = str(error) if error is not None else 'NULL'

        all_configs = get_all_configs(parallel_method, parallel_plan,
                                      alpa_global_config)
        sql_cmd = f'''INSERT INTO results VALUES
                      (
                        datetime(), {uuid},
                        {model_spec_sql_val},
                        {cluster_spec_sql_val},
                        {training_spec_sql_val},
                        {parallel_spec_sql_val},
                        {avg_lat}, {med_lat}, {p90_lat}, {max_memory},
                        {error},
                        {all_configs}
                      )'''

        self.cur.execute(sql_cmd)

        # Save (commit) the changes
        self.con.commit()