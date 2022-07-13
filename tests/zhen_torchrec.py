from enum import Enum
from typing import List, Optional, Tuple, Union, Callable
from collections import namedtuple
import logging
import time

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_frontend.test_zhen import ZHENCollection, TokenMixer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=
    '%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# A tuple to wrap all training states.
TrainState = namedtuple("TrainState", ["params", "optim_state"])


class TrainWrapper(nn.Module):
    """nn.Module to wrap model to use with train_pipeline."""

    def __init__(self, module: nn.Module, loss_func) -> None:
        super().__init__()
        self.model = module
        self.loss_func = loss_func

    def forward(self, batch) -> torch.Tensor:
        inputs, targets = batch
        out = self.model(inputs)
        return self.loss_func(out, targets)


# Define one gradient descent step
def train_step(model, optim, batch):
    # compute loss
    loss = model(batch)

    # compute gradient
    loss.backward()

    # do optimizer step
    optim.step()

    return loss


def train(model, dataloader, optim):
    latencies = []

    # Run training loops
    sstart = time.perf_counter()
    for i, batch in enumerate(dataloader):
        start = time.perf_counter()
        loss_value = train_step(model, optim, batch)
        # torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append(end - start)

        # do whatever with the loss value, e.g. plot it on a graph
        logger.info(f"Iter: {i}, Loss: {float(loss_value):.6f}")
    eend = time.perf_counter()

    print('avg:', (eend - sstart) / len(dataloader))
    latencies.sort()
    avg_s = sum(latencies) / len(dataloader)
    logger.info(f"duration of each iteration avg: {avg_s:.6f} secs, "
                f"median: {latencies[int(len(latencies) * 0.5)]} secs, "
                f"90P: {latencies[int(len(latencies) * 0.9)]} secs, "
                f"99P: {latencies[int(len(latencies) * 0.99)]} secs")


def train_zhen_homogeneous(batch_size_per_device, num_iters):
    B = batch_size_per_device
    F = 512
    D = 160
    LAYERS = 4
    OUTPUT_PER_ENSEMBLE = 32
    TOKENS = [
        TokenMixer.ATTENTION, TokenMixer.LINEAR, TokenMixer.ATTENTION,
        TokenMixer.DOT
    ]
    # B = batch_size_per_device
    # F = 37
    # D = 160
    # LAYERS = 3
    # OUTPUT_PER_ENSEMBLE = 48
    # TOKENS = [[TokenMixer.ATTENTION, TokenMixer.DOT],
    #       [
    #           TokenMixer.ATTENTION, TokenMixer.CONVOLUTION,
    #           TokenMixer.DOT
    #       ], [TokenMixer.ATTENTION, TokenMixer.DOT]]  # 3-layer ZHEN

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    zhen_module = ZHENCollection(LAYERS, D, TOKENS, F, OUTPUT_PER_ENSEMBLE)
    # print('zhen_module parameters:', sum(p.numel() for p in zhen_module.parameters() if p.requires_grad))
    # 173059456
    # 692MB

    dataloader = [(torch.empty(
        B, D, F).cuda(), torch.empty(B, D * LAYERS * OUTPUT_PER_ENSEMBLE).cuda())] * num_iters

    loss_func = lambda *args, **kwargs: torch.nn.functional.mse_loss(
        *args, **kwargs)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    train_model = TrainWrapper(zhen_module, loss_func).to(rank % 8)

    model = DistributedDataParallel(train_model, device_ids=[rank % 8])
    # model = train_model.to('cuda:0')

    print(model)
    optim = torch.optim.Adam(list(model.parameters()), lr=1e-3)
    train(model, dataloader, optim)


SETTING = {
    "batch_size_per_device": 1024,
    "num_iters": 100,
}

def main():
    batch_size_per_device = SETTING["batch_size_per_device"]
    num_iters = SETTING["num_iters"]
    train_zhen_homogeneous(batch_size_per_device, num_iters)


if __name__ == '__main__':
    import os
    os.environ["FI_PROVIDER"] = "efa"
    os.environ["FI_EFA_USE_DEVICE_RDMA"] = "1"
    os.environ["NCCL_DEBUG"] = "info"
    os.environ["LD_LIBRARY_PATH"] = os.getenv("LD_LIBRARY_PATH", "")
    print(os.environ)
    main()