from typing import Any, Optional, Callable, Tuple
import threading
import dataclasses
from dataclasses import dataclass
import threading
import subprocess
import shlex
import time

import ray
import alpa
from serde import serde
from log import logger


@serde
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

    def to_sql_values(self) -> str:
        return f'{self.num_hosts}, {self.num_devices_per_host}, "{self.transport}"'


HEAD = "a8"
FOLLOWERS = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
RAY_PATH = "/home/cjr/miniconda3/envs/alpa/bin/ray"


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
        logger.debug(cmd)
        ths.append(threading.Thread(target=run_task, args=(cmd,)))


def wait_for_threads(ths):
    logger.debug("starting threads")
    for th in ths:
        th.start()
    logger.debug("waiting for threads to finish")
    for th in ths:
        th.join()
    logger.debug("threads finished")


# shutdown ray on all machines
def shutdown_ray_cluster():
    alpa.api.is_initialized = False
    logger.warn("shutting down the alpa global_cluster")
    # Just nullify the objects instead of destroying the connections, the later does not help when the NCCL communicator
    # it self hangs.
    alpa.device_mesh.global_cluster = None
    alpa.device_mesh.global_physical_mesh = None
    alpa.device_mesh.global_virtual_physical_mesh = None
    # alpa.device_mesh.shutdown_global_cluster()
    logger.warn("disconnected from ray ")
    ray.shutdown()
    logger.warn("shutting down ray")

    th_leader = []
    ths_follower = []
    cmd_leader = {
        HEAD: [f"{RAY_PATH} stop --grace-period 10"],
    }
    cmd_follower = {}
    for w in FOLLOWERS:
        cmd_follower[w] = [f"{RAY_PATH} stop --grace-period 10"]

    # submit the kill command to leader and follower nearly the same time (This is tricky, don't know else doesn't work)
    ssh_submit(cmd_follower, ths_follower)
    ssh_submit(cmd_leader, th_leader)

    # maybe try this https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/
    # but not guaranteed to work

    wait_for_threads(ths_follower)
    wait_for_threads(th_leader)
    logger.warn("shutting down ray finishes")


def start_ray_cluster(cluster_spec: ClusterSpec):
    assert cluster_spec.num_hosts <= len(
        FOLLOWERS) + 1 and cluster_spec.num_hosts > 0
    # shutdown cluster first
    shutdown_ray_cluster()

    time.sleep(1)
    th_leader = []
    ths_follower = []
    cmd_leader = {
        HEAD: [
            f"{RAY_PATH} start --head --dashboard-host 0.0.0.0 --num-gpus {cluster_spec.num_devices_per_host}"
        ],
    }
    cmd_follower = {}
    for i in range(1, cluster_spec.num_hosts):
        cmd_follower[FOLLOWERS[i - 1]] = [
            f"{RAY_PATH} start --num-gpus {cluster_spec.num_devices_per_host} --address=a8:6379"
        ]

    ssh_submit(cmd_leader, th_leader)
    wait_for_threads(th_leader)

    ssh_submit(cmd_follower, ths_follower)
    wait_for_threads(ths_follower)

    # reset global states of alpa to make it work properly
    from alpa.timer import timers
    for timer in timers.timers.values():
        timer.reset()
    ray.init(address="auto", ignore_reinit_error=True)


def timeout_and_shutdown_ray_cluster(
    timeout_secs: int, callback: Optional[Callable[[], Any]]
) -> Tuple[threading.Event, threading.Thread]:

    def inner(e: threading.Event):
        start = time.time()
        while not e.isSet():
            time.sleep(1)
            if time.time() - start > timeout_secs:
                logger.warn("trying to shut down the ray cluster")
                if callback is not None:
                    callback()
                shutdown_ray_cluster()
                return

    e = threading.Event()
    th = threading.Thread(target=inner, args=(e,))
    th.start()
    return (e, th)