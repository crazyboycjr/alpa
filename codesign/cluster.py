import dataclasses
from dataclasses import dataclass
import threading
import subprocess
import shlex
import logging
import time

import ray
from serde import serde


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


HEAD = "a8"
FOLLOWERS = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
RAY_PATH = "/home/cjr/miniconda3/envs/alpa/bin/ray"

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

    time.sleep(1)
    th_leader = []
    ths_follower = []
    # th_checker = []
    cmd_leader = {
        HEAD: [
            f"{RAY_PATH} start --head --dashboard-host 0.0.0.0 --num-gpus {cluster_spec.num_devices_per_host}"
        ],
    }
    cmd_follower = {}
    # cmd_checker = {HEAD: [f"{RAY_PATH} status", f"{RAY_PATH} status"]}
    for i in range(1, cluster_spec.num_hosts):
        cmd_follower[FOLLOWERS[i - 1]] = [
            f"{RAY_PATH} start --num-gpus {cluster_spec.num_devices_per_host} --address=a8:6379"
        ]
    ssh_submit(cmd_leader, th_leader)
    wait_for_threads(th_leader)

    ssh_submit(cmd_follower, ths_follower)
    wait_for_threads(ths_follower)

    # ssh_submit(cmd_checker, th_checker)
    # wait_for_threads(th_checker)
    ray.init(address="auto", ignore_reinit_error=True)