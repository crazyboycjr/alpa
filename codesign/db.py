import os
from typing import Dict, Optional, Any, Sequence

import alpa
from alpa.parallel_plan import ParallelPlan
from alpa.global_env import GlobalConfig

from cluster import ClusterSpec
from training_spec import TrainingSpec

# The list of environment variable we will print
ENV_FILTER = [
    'NCCL_DEBUG', 'ALPA_USE_AWS_EFA', 'NVIDIA_TF32_OVERRIDE', 'XLA_FLAGS',
    'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'
]


def save_trial_specs(cluster_spec: ClusterSpec, model_spec: Dict[str, Any],
                     training_spec: TrainingSpec, parallel_method):
    print(cluster_spec.value_csv())
    print(model_spec)
    print(training_spec.value_csv())
    print(parallel_method)


def print_results(latencies: Sequence[float], memory: int,
                  parallel_plan: alpa.parallel_plan.ParallelPlan):
    print(latencies)
    print(memory)
    print(parallel_plan)


def save_record(cluster_spec: ClusterSpec, model_spec: Dict[str, Any],
                training_spec: TrainingSpec, parallel_method,
                alpa_global_config: GlobalConfig,
                latencies: Optional[Sequence[float]], memory: Optional[int],
                parallel_plan: Optional[ParallelPlan],
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
        print(error)