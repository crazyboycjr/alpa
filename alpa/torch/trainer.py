# pylint: disable=line-too-long, pointless-string-statement, cell-var-from-loop
"""Example trainer that runs an SGD training loop"""
import logging
import time
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
import alpa
import alpa.torch as atorch
"""
FAQ: When to use atorch vs. torch?

Answer:
- All `atorch` usage is contained within the trainer code (i.e. this file),
no `atorch` mentions in user code (e.g. test_torch_simple.py).
- No `torch` usage in trainer code. e.g. PyTorch dataloader will be
encapsulated in alpa.torch dataloader (TBD), where we will add features
related to dist dataloading.
"""

# A tuple to wrap all training states.
TrainState = namedtuple("TrainState", ["params", "bufs", "optim_state"])

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=
    '%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def train_torch_module(pt_module_gen, weight_init_func, dataloader, loss_func,
                       optim_gen, parallel_method):
    # for mode in ["local", "dist"]:
    for mode in ["dist"]:
        # "local": pure PT eager mode on a single GPU,
        #     allows print in middle of graph, no dist training
        # "dist": graph mode by lowering PT program to JAX,
        #     doesn't allow print, supports dist training
        # NOTE: as we see below, the two modes can share most of the code.
        atorch.set_mode(mode)

        # Prints verbose log for debugging.
        atorch.debug = False

        if atorch.mode() == "dist":
            alpa.init(cluster="ray")

        # Functionalize the PyTorch model and optimizer
        pt_module = atorch.meta_init(pt_module_gen)
        module_func, params_aval, bufs_aval, name_map = atorch.functionalize(
            pt_module)
        optim_func, optim_state_init_func, optim_state_aval = optim_gen(
            params_aval)

        # Define one gradient descent step
        def train_step(state, batch):
            inputs, targets = batch

            # wrap forward pass + loss computation in a function
            def compute_loss(params, bufs, inputs, targets):
                # do forward pass
                bufs, out = module_func(params, bufs, inputs)

                # do loss computation
                loss_value = loss_func(out, targets)
                return loss_value, bufs

            # do model forward + backward pass
            (loss_value, bufs), params_grad = atorch.value_and_grad(
                compute_loss, has_aux=True)(state.params, state.bufs, inputs,
                                            targets)

            # do optimizer step
            params, optim_state = optim_func(state.params, state.optim_state,
                                             params_grad)

            return TrainState(params, bufs, optim_state), loss_value

        # Define the state initialization function
        def create_train_state():
            params, bufs, optim_state = atorch.initialize_with_zeros(
                params_aval, bufs_aval, optim_state_aval)
            params, bufs = weight_init_func(pt_module, name_map, params, bufs)
            optim_state = optim_state_init_func(optim_state)
            return TrainState(params, bufs, optim_state)

        # Parallelize train function and state initialization function
        if atorch.mode() == "dist":
            train_step = alpa.parallelize(
                atorch.enable_dist_for_func(train_step),
                method=parallel_method,
                # NOTE: preserves mem addr and sharding spec for the first argument
                donate_argnums=(0,),
                # NOTE: the second argument is input batch
                batch_argnums=(1,),
                static_argnums=(),
            )

            # Assume we have a dataloader that supports `peek` function
            # (i.e. look at next batch but don't advance the pointer)
            pt_batch = atorch.make_shaped_array_from_pt_tensor(dataloader[0])

            create_train_state = alpa.parallelize(
                atorch.enable_dist_for_func(create_train_state),
                method=alpa.CreateStateParallel(train_step, pt_batch))

        # print hlo_text for debuging
        # executable = create_train_state.get_executable()
        # hlo_text = executable.get_hlo_text()
        # with open("/tmp/f.hlo", "w") as fout:
        #     fout.write(hlo_text)
        # raise NotImplementedError

        create_state_executable = create_train_state.get_executable()
        train_step_executable = train_step.get_last_executable()

        parallel_plan = train_step_executable.get_parallel_plan()
        print('get_parallel_plan', parallel_plan)
        batch_placement_specs = train_step_executable.get_input_placement_specs()[1]
        print('batch_placement_specs:', batch_placement_specs)

        # Initialize weights and optimizer states
        state = create_train_state()

        latencies = []
        # Run training loops
        for i, pt_batch in enumerate(dataloader):
            pt_batch = atorch.to_format(atorch.mode(), pt_batch)
            # print('pt_batch:', pt_batch[0], dir(pt_batch[0]), type(pt_batch[0]), pt_batch[0].shape, pt_batch[0].size)

            train_step_executable.sync()
            # start = time.perf_counter()
            state, loss_value = train_step(state, pt_batch)
            train_step_executable.sync()
            # end = time.perf_counter()
            # latencies.append(end - start)

            if (i + 1) % 5 == 0:
                # print every 5 iterations
                print(train_step_executable.get_execution_time_costs()[2:])

            # do whatever with the loss value, e.g. plot it on a graph
            logger.info(f"Iter: {i}, Loss: {float(loss_value):.6f}")
        latencies = train_step_executable.get_execution_time_costs()[2:]

        latencies.sort()
        print("latencies:", latencies)
        avg_s = sum(latencies) / len(dataloader)
        logger.info(f"duration of each iteration avg: {avg_s:.6f} secs, "
                    f"median: {latencies[int(len(latencies) * 0.5)]} secs, "
                    f"90P: {latencies[int(len(latencies) * 0.9)]} secs, "
                    f"99P: {latencies[int(len(latencies) * 0.99)]} secs")

        if isinstance(parallel_method, alpa.PipeshardParallel):
            max_mem_allocated = train_step_executable.mesh_group.get_max_memory_allocated()
        else:
            max_mem_allocated = train_step_executable.physical_mesh.get_max_memory_allocated()
        print(f"max_mem_allocated: {max_mem_allocated}")

        if atorch.mode() == "dist":
            alpa.shutdown()

        return latencies, max_mem_allocated, parallel_plan