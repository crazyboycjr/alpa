#!/usr/bin/env bash

# do not retry-failed cases
CMD="env CODESIGN_LOG_LEVEL=info NCCL_DEBUG=info \
    /home/cjr/miniconda3/envs/alpa/bin/python codesign/main.py \
    --manual-job-timeout 600"

while true; do
    ${CMD}
    if [ $? -eq 0 ]; then
        break
    fi
    echo "retrying..."
    sleep 1
done