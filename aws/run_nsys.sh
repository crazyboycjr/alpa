#!/usr/bin/env bash

# Profile DHNE with nsight-systems or nsight-compute

function run_help() { #HELP Display this message:\nZHEN help
    sed -n "s/^.*#HELP\\s//p;" < "$1" | sed "s/\\\\n/\n\t/g;s/$/\n/;s!ZHEN!${1/!/\\!}!g"
    exit 0
}

function run_nsys() { #HELP Profile ZHEN with nsight-systems:\nZHEN nsys
    # /usr/local/cuda/nsight-systems-2021.3.2/bin/nsys
    nsys profile \
        --export=json \
        -t cuda,nvtx,osrt,cudnn,cublas \
        --cuda-memory-usage=true \
        "$@"
}

function run_ncu() { #HELP Profile ZHEN with nsight-compute:\nZHEN ncu
    # /usr/local/cuda/nsight-compute-2021.2.2/ncu
    CUDA_INJECTION64_PATH=none \
    LD_LIBRARY_PATH=/usr/local/cuda/nsight-compute-2021.2.2/target/linux-desktop-glibc_2_11_3-x64:$LD_LIBRARY_PATH \
    ncu \
        --target-processes all \
        --set full \
        --nvtx \
        -o report -f --import-source yes \
        "$@"
}

function run_local() { #HELP Run ZHEN in local mode:\nZHEN local
    "$@"
}

[[ -z "${1-}" ]] && run_help "$0"
case $1 in
    local|nsys|ncu) CMD=run_"$*" ;;
    *) run_help "$0" ;;
esac

$CMD