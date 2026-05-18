pkill -9 python
set -x

export PYTHONPATH=/home/ma-user/work/algorithm/lcc_verl_clone/Verl:$PYTHONPATH
export VERL_DATA_FOLDER=/cache/data/taco
export HF_MODEL_FOLDER=/cache
export VLLM_ASCEND_ENABLE_NZ=0

# export VLLM_ATTENTION_BACKEND=XFORMERS

export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
#规避ray在device侧调用无法根据is_npu_available接口识别设备可用性
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
#根据当前设备和需要卡数定义
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# #使能推理EP时需要
# export SGLANG_DEEPEP_BF16_DISPATCH=1
# export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export VERL_PROJECT_NAME='conf_example_taco'
export VERL_EXPERIMENT_NAME='qwen25_7b'
export VERL_FILE_LOGGER_ROOT=/cache/log
export TENSORBOARD_DIR=/cache/tensorboard/$VERL_PROJECT_NAME/${VERL_EXPERIMENT_NAME}


python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=acc_conf \
    data.train_files=$VERL_DATA_FOLDER/train.parquet \
    data.val_files=$VERL_DATA_FOLDER/test.parquet \
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
    +reward_model.reward_kwargs.n_samples_per_prompt=16 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard','file'] \
    trainer.project_name=$VERL_PROJECT_NAME \
    trainer.experiment_name=$VERL_EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.default_local_dir=/cache/code/model_new \
    trainer.validation_data_dir=$VERL_FILE_LOGGER_ROOT/val_log \
    trainer.test_freq=40 \
    trainer.total_epochs=1 2>&1 | tee "$VERL_FILE_LOGGER_ROOT/$(date +"%m%d_%H%M").log"





