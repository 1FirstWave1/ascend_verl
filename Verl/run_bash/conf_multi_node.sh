cd Verl
#任务启动需要安装
pip install pyext

pkill -9 python
ray stop --force
set -x

#不同worker信息去重与否
export RAY_DEDUP_LOGS=1
#Hydra报错是否打印完整日志
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/opt/huawei/schedule-train/algorithm/Verl:$PYTHONPATH
export VERL_DATA_FOLDER=/opt/huawei/dataset/lcc_guiyang
export HF_MODEL_FOLDER=/opt/huawei/dataset/lcc_guiyang
#不进行acend上的优化
export VLLM_ASCEND_ENABLE_NZ=0
#加载指定动态库
export LD_PRELOAD=${LD_PRELOAD:-/usr/local/lib/libjemalloc.so.2}

ulimit -n 32768

export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HCCL_IF_BASE_PORT=64000
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export VERL_PROJECT_NAME='grpo_taco'
export VERL_EXPERIMENT_NAME='qwen25_7b'
export VERL_FILE_LOGGER_ROOT=$VERL_DATA_FOLDER/code/log
export TENSORBOARD_DIR=$VERL_DATA_FOLDER/code/tensorboard/$VERL_PROJECT_NAME/${VERL_EXPERIMENT_NAME}

NNODES=${MA_NUM_HOSTS:-4}
NPUS_PER_NODE=${MA_NUM_GPUS:-8}
WORLD_SIZE=$((NNODES * NPUS_PER_NODE))
NODE_RANK=${VC_TASK_INDEX:-0}
MASTER_RANK=${MASTER_RANK:-0}

export WORLD_SIZE

MASTER_ADDR_DEFAULT="${MA_VJ_NAME:-localhost}-${MA_TASK_NAME:-job}-${MASTER_RANK}.${MA_VJ_NAME:-local}"
CURRENT_IP_DEFAULT="${MA_VJ_NAME:-localhost}-${MA_TASK_NAME:-job}-${NODE_RANK}.${MA_VJ_NAME:-local}"

MASTER_ADDR=${MASTER_ADDR:-$MASTER_ADDR_DEFAULT}
CURRENT_IP=${CURRENT_IP:-$CURRENT_IP_DEFAULT}
RAY_PORT=${RAY_PORT:-6766}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8260}

echo "**** NNODES: ${NNODES}"
echo "**** NPUS_PER_NODE: ${NPUS_PER_NODE}"
echo "**** WORLD_SIZE: ${WORLD_SIZE}"
echo "**** NODE_RANK: ${NODE_RANK}"
echo "**** MASTER_ADDR: ${MASTER_ADDR}"
echo "**** CURRENT_IP: ${CURRENT_IP}"

start_training() {
  python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$VERL_DATA_FOLDER/code/taco/train.parquet \
    data.val_files=$VERL_DATA_FOLDER/code/taco/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_FOLDER/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.use_reward_loop=False \
    reward_model.reward_manager=acc_conf \
    reward_model.reward_kwargs.n_samples_per_prompt=16 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard','file'] \
    trainer.project_name=$VERL_PROJECT_NAME \
    trainer.experiment_name=$VERL_EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=40 \
    trainer.default_local_dir=$VERL_DATA_FOLDER/code/model_new \
    trainer.validation_data_dir=$VERL_FILE_LOGGER_ROOT/val_log \
    trainer.test_freq=40 \
    trainer.total_epochs=1 2>&1 | tee "$VERL_FILE_LOGGER_ROOT/$(date +"%m%d_%H%M").log"
}

if [ "$NODE_RANK" -eq "$MASTER_RANK" ]; then
  ray start \
    --head \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --node-ip-address="${CURRENT_IP}" \
    --resources="{\"NPU\": ${NPUS_PER_NODE}}"

  while true; do
    ray_status_output=$(ray status)
    npu_total=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+(\.\d+)?(?=\s*NPU)' | head -n 1)
    if [ -z "${npu_total}" ]; then
      device_count=0
    else
      npu_count_int=$(echo "$npu_total" | awk '{print int($1)}')
      device_count=$((npu_count_int / NPUS_PER_NODE))
    fi

    if [ "$device_count" -ge "$NNODES" ]; then
      echo "Ray cluster is ready with ${device_count} nodes."
      ray status
      start_training
      break
    else
      echo "Waiting for Ray cluster. Expected ${NNODES} nodes, current ${device_count}."
      sleep 5
    fi
  done
else
  while true; do
    ray start \
      --address="${MASTER_ADDR}:${RAY_PORT}" \
      --node-ip-address="${CURRENT_IP}" \
      --resources="{\"NPU\": ${NPUS_PER_NODE}}"

    if ray status >/dev/null 2>&1; then
      echo "Successfully connected to Ray cluster."
      break
    else
      echo "Failed to connect to Ray cluster. Retrying in 5 seconds..."
      ray stop --force
      sleep 5
    fi
  done
fi

echo "**** END ****"
sleep 600



