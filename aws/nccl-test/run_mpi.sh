#!/usr/bin/env bash

WORKDIR=$(dirname `realpath $0`)
echo $WORKDIR

/opt/amazon/openmpi/bin/mpirun \
    -x FI_EFA_FORK_SAFE=1 \
    -x FI_PROVIDER="efa" \
    -x FI_EFA_USE_DEVICE_RDMA=1 \
    -x LD_LIBRARY_PATH=/opt/amazon/efa/lib:${CUDA_HOME}/efa/lib:$LD_LIBRARY_PATH \
    -x NCCL_DEBUG=info \
    --hostfile ~/nfs/hostfile -n 2 -N 1 \
    --mca btl_tcp_if_exclude lo,docker0 --bind-to numa \
    ${WORKDIR}/send_recv

    # ${WORKDIR}/send_recv

    # -x CUDA_VISIBLE_DEVICES=0 \
    # -x NCCL_NET=Socket \
    # -x NCCL_DEBUG_SUBSYS=GRAPH,TUNING,INIT \
    # -x NCCL_TOPO_DUMP_FILE=system.xml \
    # -x NCCL_IGNORE_CPU_AFFINITY=1 \ not helpful
    # /opt/amazon/openmpi/lib:
    # -x FI_EFA_USE_DEVICE_RDMA=1 \
    # -x NCCL_NET_GDR_LEVEL=0 \ not helpful
    # -x NCCL_NET_GDR_READ=0 \ not helpful
    # -x NCCL_MIN_NCHANNELS=32 \ This is useful when each node only use 1 GPU
    # -x CUDA_VISIBLE_DEVICES="0" \
    # --mca btl tcp,self
    # -x NCCL_P2P_LEVEL=3 \
    # --mca pml ^cm \
    # -x FI_LOG_LEVEL="debug" \
    # -x FI_PROVIDER="tcp" \
    # -x FI_SOCKETS_IFACE="ens32" \
    # -x NCCL_CROSS_NIC="1" \ does not help utilizing multiple NICs
    # -x NCCL_ALGO=ring \ ring is slower
    # --mca pml ^cm \
    # -x NCCL_COLLNET_ENABLE="0" \
    # -x NCCL_PROTO=simple \
    # -x NCCL_BUFFSIZE=1048576 \ don't change the default 4M