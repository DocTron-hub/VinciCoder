#!/bin/bash



# NOTE:
# This script launches a distributed training job.
# Please modify the following three path/name variables to match your setup.
# Please pay attention to trainer.nnodes only for GRPO and worker.reward.num_gpus should utilize other gpus
# =========================================================================

set -x

export PYTHONUNBUFFERED=1
MODEL_PATH=Qwen3-VL # replace it with your local file path
EXP_NAME=Vincicoder-8B # replace it with your exp name
DATA_FOLDER=ViRL # replace it with your local data folder path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/zhaoxuanle/EasyR1/dataset/ \
    data.rollout_batch_size=256 \
    data.mini_rollout_batch_size=256 \
    data.format_prompt=./examples/format_prompt/vision.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    algorithm.disable_kl=True \
    worker.rollout.max_num_batched_tokens=22528 \
    worker.rollout.gpu_memory_utilization=0.35 \
    worker.rollout.tensor_parallel_size=4 \
    worker.actor.global_batch_size=256 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.reward.reward_function=./examples/reward_function/vincicoder.py:compute_score \
    worker.reward.num_cpus=80\
    worker.reward.num_gpus=4\
    trainer.experiment_name=${EXP_NAME} \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    trainer.save_limit=30 \
    trainer.save_checkpoint_path=./checkpoints/${EXP_NAME} \
