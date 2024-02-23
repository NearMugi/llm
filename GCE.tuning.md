```log
/content/LLaMA-Factory
2024-02-15 23:25:51.140989: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-02-15 23:25:51.141045: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-02-15 23:25:51.142980: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-02-15 23:25:52.384245: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://c179f2beffa9ba70dc.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
02/15/2024 23:26:52 - WARNING - llmtuner.hparams.parser - We recommend enable `upcast_layernorm` in quantized training.
02/15/2024 23:26:52 - WARNING - llmtuner.hparams.parser - `ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.
[INFO|training_args.py:1828] 2024-02-15 23:26:52,957 >> PyTorch: setting up devices
/usr/local/lib/python3.10/dist-packages/transformers/training_args.py:1741: FutureWarning: `--push_to_hub_token` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--hub_token` instead.
  warnings.warn(
02/15/2024 23:26:52 - INFO - llmtuner.hparams.parser - Process rank: 0, device: cuda:0, n_gpu: 1
  distributed training: True, compute dtype: torch.float16
02/15/2024 23:26:52 - INFO - llmtuner.hparams.parser - Training/evaluation parameters Seq2SeqTrainingArguments(
_n_gpu=1,
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
bf16=False,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=False,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=True,
dispatch_batches=None,
do_eval=False,
do_predict=False,
do_train=True,
eval_accumulation_steps=None,
eval_delay=0,
eval_steps=None,
evaluation_strategy=no,
fp16=True,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
generation_config=None,
generation_max_length=None,
generation_num_beams=None,
gradient_accumulation_steps=4,
gradient_checkpointing=False,
gradient_checkpointing_kwargs=None,
greater_is_better=None,
group_by_length=False,
half_precision_backend=auto,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=False,
hub_strategy=every_save,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_inputs_for_metrics=False,
include_num_input_tokens_seen=False,
include_tokens_per_second=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=0.0002,
length_column_name=length,
load_best_model_at_end=False,
local_rank=0,
log_level=passive,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=saves/InternLM-7B/lora/train_2024-02-15-23-26-12/runs/Feb15_23-26-52_0bb75e04839a,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=5,
logging_strategy=steps,
lr_scheduler_kwargs={},
lr_scheduler_type=cosine,
max_grad_norm=0.3,
max_steps=-1,
metric_for_best_model=None,
mp_parameters=,
neftune_noise_alpha=None,
no_cuda=False,
num_train_epochs=3.0,
optim=adamw_torch,
optim_args=None,
output_dir=saves/InternLM-7B/lora/train_2024-02-15-23-26-12,
overwrite_output_dir=False,
past_index=-1,
per_device_eval_batch_size=8,
per_device_train_batch_size=16,
predict_with_generate=False,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=['tensorboard'],
resume_from_checkpoint=None,
run_name=saves/InternLM-7B/lora/train_2024-02-15-23-26-12,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=100,
save_strategy=steps,
save_total_limit=None,
seed=42,
skip_memory_metrics=True,
sortish_sampler=False,
split_batches=False,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_cpu=False,
use_ipex=False,
use_legacy_prediction_loop=False,
use_mps_device=False,
warmup_ratio=0.0,
warmup_steps=0,
weight_decay=0.0,
)
[INFO|tokenization_utils_base.py:2027] 2024-02-15 23:26:53,357 >> loading file tokenizer.model from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer.model
[INFO|tokenization_utils_base.py:2027] 2024-02-15 23:26:53,357 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2027] 2024-02-15 23:26:53,357 >> loading file special_tokens_map.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/special_tokens_map.json
[INFO|tokenization_utils_base.py:2027] 2024-02-15 23:26:53,357 >> loading file tokenizer_config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer_config.json
[INFO|tokenization_utils_base.py:2027] 2024-02-15 23:26:53,357 >> loading file tokenizer.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer.json
[INFO|configuration_utils.py:729] 2024-02-15 23:26:53,823 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-15 23:26:53,824 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

02/15/2024 23:26:53 - INFO - llmtuner.model.patcher - Quantizing model to 4 bit.
[INFO|modeling_utils.py:3476] 2024-02-15 23:26:54,396 >> loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/pytorch_model.bin.index.json
[INFO|modeling_utils.py:1426] 2024-02-15 23:26:54,509 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
[INFO|configuration_utils.py:826] 2024-02-15 23:26:54,510 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0
}

[INFO|modeling_utils.py:3615] 2024-02-15 23:27:01,731 >> Detected 4-bit loading: activating 4-bit loading for this model
Loading checkpoint shards:   0% 0/2 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
Loading checkpoint shards: 100% 2/2 [02:29<00:00, 74.72s/it]
[INFO|modeling_utils.py:4350] 2024-02-15 23:29:31,471 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

[INFO|modeling_utils.py:4358] 2024-02-15 23:29:31,471 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at elyza/ELYZA-japanese-Llama-2-7b-instruct.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
[INFO|configuration_utils.py:781] 2024-02-15 23:29:31,643 >> loading configuration file generation_config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/generation_config.json
[INFO|configuration_utils.py:826] 2024-02-15 23:29:31,644 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "use_cache": false
}

02/15/2024 23:29:32 - INFO - llmtuner.model.patcher - Gradient checkpointing enabled.
02/15/2024 23:29:32 - INFO - llmtuner.model.adapter - Fine-tuning method: LoRA
02/15/2024 23:29:32 - INFO - llmtuner.model.loader - trainable params: 4194304 || all params: 6742609920 || trainable%: 0.0622
02/15/2024 23:29:32 - INFO - llmtuner.data.loader - Loading dataset /content/databricks-dolly-15k-ja-near-mugi.json...
02/15/2024 23:29:32 - WARNING - llmtuner.data.utils - Checksum failed: missing SHA-1 hash value in dataset_info.json.
Using custom data configuration default-9b3a6747b0dee9f3
Loading Dataset Infos from /usr/local/lib/python3.10/dist-packages/datasets/packaged_modules/json
Overwrite dataset info from restored data version if exists.
Loading Dataset info from /root/.cache/huggingface/datasets/json/default-9b3a6747b0dee9f3/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96
Found cached dataset json (/root/.cache/huggingface/datasets/json/default-9b3a6747b0dee9f3/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96)
Loading Dataset info from /root/.cache/huggingface/datasets/json/default-9b3a6747b0dee9f3/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96
Loading cached processed dataset at /root/.cache/huggingface/datasets/json/default-9b3a6747b0dee9f3/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96/cache-3f4a5a914c232742.arrow
02/15/2024 23:29:33 - INFO - llmtuner.data.loader - Loading dataset /content/databricks-dolly-15k-ja-nya.json...
02/15/2024 23:29:33 - WARNING - llmtuner.data.utils - Checksum failed: missing SHA-1 hash value in dataset_info.json.
Using custom data configuration default-3446eb7812b283cd
Loading Dataset Infos from /usr/local/lib/python3.10/dist-packages/datasets/packaged_modules/json
Overwrite dataset info from restored data version if exists.
Loading Dataset info from /root/.cache/huggingface/datasets/json/default-3446eb7812b283cd/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96
Found cached dataset json (/root/.cache/huggingface/datasets/json/default-3446eb7812b283cd/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96)
Loading Dataset info from /root/.cache/huggingface/datasets/json/default-3446eb7812b283cd/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96
Loading cached processed dataset at /root/.cache/huggingface/datasets/json/default-3446eb7812b283cd/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96/cache-7ef9c8d37296d9b3.arrow
Running tokenizer on dataset:   0% 0/10163 [00:00<?, ? examples/s]Caching processed dataset at /root/.cache/huggingface/datasets/json/default-9b3a6747b0dee9f3/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96/cache-560081fb5040a0af.arrow
Running tokenizer on dataset: 100% 10163/10163 [00:18<00:00, 564.43 examples/s]
input_ids:
[12968, 29901, 29871, 30429, 31049, 30199, 30333, 31374, 30396, 235, 187, 146, 30441, 30914, 30466, 30330, 30635, 30310, 30449, 31250, 30389, 30371, 234, 143, 174, 30499, 30427, 30412, 30882, 13, 30635, 30310, 30449, 234, 143, 174, 30499, 30427, 30267, 29906, 29900, 29900, 29947, 30470, 29941, 30534, 29947, 30325, 30486, 30441, 30553, 30199, 236, 158, 143, 30499, 30427, 30267, 29896, 233, 176, 182, 30441, 30499, 30449, 30661, 31400, 234, 143, 174, 30499, 30326, 30366, 30267, 31954, 31085, 30364, 30868, 31085, 30199, 233, 178, 158, 31085, 30499, 30427, 30267, 233, 178, 158, 230, 132, 168, 30568, 31206, 30298, 30458, 31076, 30538, 30499, 30313, 30199, 30880, 30723, 235, 139, 147, 30954, 30441, 30427, 30267, 30566, 30566, 31687, 30458, 30257, 31076, 30538, 30499, 30427, 30267, 30257, 31757, 30371, 30613, 30999, 30499, 30427, 30267, 13, 7900, 22137, 29901, 29871, 29871, 30635, 30310, 30449, 30682, 30972, 30568, 30466, 30735, 30389, 30371, 30353, 30972, 30566, 30553, 30466, 30298, 30332, 234, 143, 174, 30499, 30427, 30267, 2]
inputs:
Human: 上記の文章を踏まえて、ニアはどんな猫ですか？
ニアは猫です。2008年3月8日生まれの雌です。1歳までは野良猫でした。茶色と白色の毛色です。毛づくろいが好きで人の手も舐めます。ささ身が大好きです。大切な家族です。
Assistant:  ニアは可愛くてみんなに愛されている猫です。</s>
label_ids:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 29871, 30635, 30310, 30449, 30682, 30972, 30568, 30466, 30735, 30389, 30371, 30353, 30972, 30566, 30553, 30466, 30298, 30332, 234, 143, 174, 30499, 30427, 30267, 2]
labels:
ニアは可愛くてみんなに愛されている猫です。</s>
[INFO|training_args.py:1828] 2024-02-15 23:29:52,446 >> PyTorch: setting up devices
[INFO|trainer.py:571] 2024-02-15 23:29:52,649 >> Using auto half precision backend
[INFO|trainer.py:1721] 2024-02-15 23:29:52,981 >> ***** Running training *****
[INFO|trainer.py:1722] 2024-02-15 23:29:52,981 >>   Num examples = 10,163
[INFO|trainer.py:1723] 2024-02-15 23:29:52,981 >>   Num Epochs = 3
[INFO|trainer.py:1724] 2024-02-15 23:29:52,981 >>   Instantaneous batch size per device = 16
[INFO|trainer.py:1727] 2024-02-15 23:29:52,981 >>   Total train batch size (w. parallel, distributed & accumulation) = 64
[INFO|trainer.py:1728] 2024-02-15 23:29:52,981 >>   Gradient Accumulation steps = 4
[INFO|trainer.py:1729] 2024-02-15 23:29:52,981 >>   Total optimization steps = 477
[INFO|trainer.py:1730] 2024-02-15 23:29:52,984 >>   Number of trainable parameters = 4,194,304
02/15/2024 23:31:47 - INFO - llmtuner.extras.callbacks - {'loss': 1.3183, 'learning_rate': 1.9995e-04, 'epoch': 0.03}
{'loss': 1.3183, 'learning_rate': 0.00019994578321597258, 'epoch': 0.03}
02/15/2024 23:33:35 - INFO - llmtuner.extras.callbacks - {'loss': 1.2233, 'learning_rate': 1.9978e-04, 'epoch': 0.06}
{'loss': 1.2233, 'learning_rate': 0.0001997831916530837, 'epoch': 0.06}
02/15/2024 23:35:23 - INFO - llmtuner.extras.callbacks - {'loss': 1.0804, 'learning_rate': 1.9951e-04, 'epoch': 0.09}
{'loss': 1.0804, 'learning_rate': 0.0001995124016151664, 'epoch': 0.09}
02/15/2024 23:37:11 - INFO - llmtuner.extras.callbacks - {'loss': 1.0266, 'learning_rate': 1.9913e-04, 'epoch': 0.13}
{'loss': 1.0266, 'learning_rate': 0.0001991337067295207, 'epoch': 0.13}
02/15/2024 23:38:59 - INFO - llmtuner.extras.callbacks - {'loss': 0.9723, 'learning_rate': 1.9865e-04, 'epoch': 0.16}
{'loss': 0.9723, 'learning_rate': 0.00019864751762852317, 'epoch': 0.16}
02/15/2024 23:40:47 - INFO - llmtuner.extras.callbacks - {'loss': 1.0440, 'learning_rate': 1.9805e-04, 'epoch': 0.19}
{'loss': 1.044, 'learning_rate': 0.00019805436150436352, 'epoch': 0.19}
02/15/2024 23:42:35 - INFO - llmtuner.extras.callbacks - {'loss': 0.9885, 'learning_rate': 1.9735e-04, 'epoch': 0.22}
{'loss': 0.9885, 'learning_rate': 0.00019735488153739127, 'epoch': 0.22}
02/15/2024 23:44:23 - INFO - llmtuner.extras.callbacks - {'loss': 1.0006, 'learning_rate': 1.9655e-04, 'epoch': 0.25}
{'loss': 1.0006, 'learning_rate': 0.00019654983619869242, 'epoch': 0.25}
02/15/2024 23:46:11 - INFO - llmtuner.extras.callbacks - {'loss': 0.9925, 'learning_rate': 1.9564e-04, 'epoch': 0.28}
{'loss': 0.9925, 'learning_rate': 0.00019564009842765225, 'epoch': 0.28}
02/15/2024 23:47:59 - INFO - llmtuner.extras.callbacks - {'loss': 0.9828, 'learning_rate': 1.9463e-04, 'epoch': 0.31}
{'loss': 0.9828, 'learning_rate': 0.00019462665468539584, 'epoch': 0.31}
02/15/2024 23:49:46 - INFO - llmtuner.extras.callbacks - {'loss': 1.0016, 'learning_rate': 1.9351e-04, 'epoch': 0.35}
{'loss': 1.0016, 'learning_rate': 0.00019351060388513304, 'epoch': 0.35}
02/15/2024 23:51:34 - INFO - llmtuner.extras.callbacks - {'loss': 0.9784, 'learning_rate': 1.9229e-04, 'epoch': 0.38}
{'loss': 0.9784, 'learning_rate': 0.00019229315620056803, 'epoch': 0.38}
02/15/2024 23:53:22 - INFO - llmtuner.extras.callbacks - {'loss': 0.9474, 'learning_rate': 1.9098e-04, 'epoch': 0.41}
{'loss': 0.9474, 'learning_rate': 0.0001909756317536643, 'epoch': 0.41}
02/15/2024 23:55:10 - INFO - llmtuner.extras.callbacks - {'loss': 0.9493, 'learning_rate': 1.8956e-04, 'epoch': 0.44}
{'loss': 0.9493, 'learning_rate': 0.0001895594591831896, 'epoch': 0.44}
02/15/2024 23:56:58 - INFO - llmtuner.extras.callbacks - {'loss': 0.9942, 'learning_rate': 1.8805e-04, 'epoch': 0.47}
{'loss': 0.9942, 'learning_rate': 0.00018804617409559198, 'epoch': 0.47}
02/15/2024 23:58:46 - INFO - llmtuner.extras.callbacks - {'loss': 1.0006, 'learning_rate': 1.8644e-04, 'epoch': 0.50}
{'loss': 1.0006, 'learning_rate': 0.00018643741739988673, 'epoch': 0.5}
02/16/2024 00:00:34 - INFO - llmtuner.extras.callbacks - {'loss': 0.9807, 'learning_rate': 1.8473e-04, 'epoch': 0.53}
{'loss': 0.9807, 'learning_rate': 0.0001847349335283603, 'epoch': 0.53}
02/16/2024 00:02:23 - INFO - llmtuner.extras.callbacks - {'loss': 0.9455, 'learning_rate': 1.8294e-04, 'epoch': 0.57}
{'loss': 0.9455, 'learning_rate': 0.0001829405685450202, 'epoch': 0.57}
02/16/2024 00:04:11 - INFO - llmtuner.extras.callbacks - {'loss': 0.9566, 'learning_rate': 1.8106e-04, 'epoch': 0.60}
{'loss': 0.9566, 'learning_rate': 0.00018105626814384173, 'epoch': 0.6}
02/16/2024 00:05:59 - INFO - llmtuner.extras.callbacks - {'loss': 0.9876, 'learning_rate': 1.7908e-04, 'epoch': 0.63}
{'loss': 0.9876, 'learning_rate': 0.00017908407553898282, 'epoch': 0.63}
[INFO|trainer.py:2936] 2024-02-16 00:05:59,111 >> Saving model checkpoint to saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-100
[INFO|configuration_utils.py:729] 2024-02-16 00:05:59,493 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-16 00:05:59,494 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|tokenization_utils_base.py:2433] 2024-02-16 00:05:59,585 >> tokenizer config file saved in saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-100/tokenizer_config.json
[INFO|tokenization_utils_base.py:2442] 2024-02-16 00:05:59,585 >> Special tokens file saved in saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-100/special_tokens_map.json
02/16/2024 00:07:47 - INFO - llmtuner.extras.callbacks - {'loss': 0.9625, 'learning_rate': 1.7703e-04, 'epoch': 0.66}
{'loss': 0.9625, 'learning_rate': 0.00017702612924925376, 'epoch': 0.66}
02/16/2024 00:09:35 - INFO - llmtuner.extras.callbacks - {'loss': 0.9503, 'learning_rate': 1.7488e-04, 'epoch': 0.69}
{'loss': 0.9503, 'learning_rate': 0.00017488466077924525, 'epoch': 0.69}
02/16/2024 00:11:22 - INFO - llmtuner.extras.callbacks - {'loss': 0.9748, 'learning_rate': 1.7266e-04, 'epoch': 0.72}
{'loss': 0.9748, 'learning_rate': 0.00017266199219962797, 'epoch': 0.72}
02/16/2024 00:13:11 - INFO - llmtuner.extras.callbacks - {'loss': 1.0007, 'learning_rate': 1.7036e-04, 'epoch': 0.75}
{'loss': 1.0007, 'learning_rate': 0.00017036053362924896, 'epoch': 0.75}
02/16/2024 00:14:58 - INFO - llmtuner.extras.callbacks - {'loss': 0.9485, 'learning_rate': 1.6798e-04, 'epoch': 0.79}
{'loss': 0.9485, 'learning_rate': 0.0001679827806217533, 'epoch': 0.79}
02/16/2024 00:16:44 - INFO - llmtuner.extras.callbacks - {'loss': 0.9209, 'learning_rate': 1.6553e-04, 'epoch': 0.82}
{'loss': 0.9209, 'learning_rate': 0.0001655313114595666, 'epoch': 0.82}
02/16/2024 00:18:32 - INFO - llmtuner.extras.callbacks - {'loss': 0.8986, 'learning_rate': 1.6301e-04, 'epoch': 0.85}
{'loss': 0.8986, 'learning_rate': 0.00016300878435817113, 'epoch': 0.85}
02/16/2024 00:20:20 - INFO - llmtuner.extras.callbacks - {'loss': 0.9094, 'learning_rate': 1.6042e-04, 'epoch': 0.88}
{'loss': 0.9094, 'learning_rate': 0.0001604179345837081, 'epoch': 0.88}
02/16/2024 00:22:09 - INFO - llmtuner.extras.callbacks - {'loss': 0.9600, 'learning_rate': 1.5776e-04, 'epoch': 0.91}
{'loss': 0.96, 'learning_rate': 0.00015776157148703095, 'epoch': 0.91}
02/16/2024 00:23:57 - INFO - llmtuner.extras.callbacks - {'loss': 0.9396, 'learning_rate': 1.5504e-04, 'epoch': 0.94}
{'loss': 0.9396, 'learning_rate': 0.00015504257545742584, 'epoch': 0.94}
02/16/2024 00:25:45 - INFO - llmtuner.extras.callbacks - {'loss': 0.9414, 'learning_rate': 1.5226e-04, 'epoch': 0.97}
{'loss': 0.9414, 'learning_rate': 0.00015226389479930296, 'epoch': 0.97}
02/16/2024 00:27:29 - INFO - llmtuner.extras.callbacks - {'loss': 0.9046, 'learning_rate': 1.4943e-04, 'epoch': 1.01}
{'loss': 0.9046, 'learning_rate': 0.0001494285425352448, 'epoch': 1.01}
02/16/2024 00:29:17 - INFO - llmtuner.extras.callbacks - {'loss': 0.9159, 'learning_rate': 1.4654e-04, 'epoch': 1.04}
{'loss': 0.9159, 'learning_rate': 0.00014653959313887813, 'epoch': 1.04}
02/16/2024 00:31:05 - INFO - llmtuner.extras.callbacks - {'loss': 0.9068, 'learning_rate': 1.4360e-04, 'epoch': 1.07}
{'loss': 0.9068, 'learning_rate': 0.0001436001792011128, 'epoch': 1.07}
02/16/2024 00:32:53 - INFO - llmtuner.extras.callbacks - {'loss': 0.9078, 'learning_rate': 1.4061e-04, 'epoch': 1.10}
{'loss': 0.9078, 'learning_rate': 0.00014061348803336135, 'epoch': 1.1}
02/16/2024 00:34:41 - INFO - llmtuner.extras.callbacks - {'loss': 0.9534, 'learning_rate': 1.3758e-04, 'epoch': 1.13}
{'loss': 0.9534, 'learning_rate': 0.00013758275821142382, 'epoch': 1.13}
02/16/2024 00:36:29 - INFO - llmtuner.extras.callbacks - {'loss': 0.9334, 'learning_rate': 1.3451e-04, 'epoch': 1.16}
{'loss': 0.9334, 'learning_rate': 0.00013451127606378425, 'epoch': 1.16}
02/16/2024 00:38:17 - INFO - llmtuner.extras.callbacks - {'loss': 0.9161, 'learning_rate': 1.3140e-04, 'epoch': 1.19}
{'loss': 0.9161, 'learning_rate': 0.0001314023721081274, 'epoch': 1.19}
02/16/2024 00:40:05 - INFO - llmtuner.extras.callbacks - {'loss': 0.9066, 'learning_rate': 1.2826e-04, 'epoch': 1.23}
{'loss': 0.9066, 'learning_rate': 0.0001282594174399399, 'epoch': 1.23}
02/16/2024 00:41:53 - INFO - llmtuner.extras.callbacks - {'loss': 0.8987, 'learning_rate': 1.2509e-04, 'epoch': 1.26}
{'loss': 0.8987, 'learning_rate': 0.00012508582007711075, 'epoch': 1.26}
[INFO|trainer.py:2936] 2024-02-16 00:41:53,177 >> Saving model checkpoint to saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-200
[INFO|configuration_utils.py:729] 2024-02-16 00:41:53,499 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-16 00:41:53,500 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|tokenization_utils_base.py:2433] 2024-02-16 00:41:53,555 >> tokenizer config file saved in saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-200/tokenizer_config.json
[INFO|tokenization_utils_base.py:2442] 2024-02-16 00:41:53,556 >> Special tokens file saved in saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-200/special_tokens_map.json
02/16/2024 00:43:41 - INFO - llmtuner.extras.callbacks - {'loss': 0.9299, 'learning_rate': 1.2189e-04, 'epoch': 1.29}
{'loss': 0.9299, 'learning_rate': 0.00012188502126449615, 'epoch': 1.29}
02/16/2024 00:45:29 - INFO - llmtuner.extras.callbacks - {'loss': 0.9166, 'learning_rate': 1.1866e-04, 'epoch': 1.32}
{'loss': 0.9166, 'learning_rate': 0.00011866049174245491, 'epoch': 1.32}
02/16/2024 00:47:17 - INFO - llmtuner.extras.callbacks - {'loss': 0.9111, 'learning_rate': 1.1542e-04, 'epoch': 1.35}
{'loss': 0.9111, 'learning_rate': 0.00011541572798340074, 'epoch': 1.35}
02/16/2024 00:49:05 - INFO - llmtuner.extras.callbacks - {'loss': 0.9285, 'learning_rate': 1.1215e-04, 'epoch': 1.38}
{'loss': 0.9285, 'learning_rate': 0.00011215424840045255, 'epoch': 1.38}
02/16/2024 00:50:53 - INFO - llmtuner.extras.callbacks - {'loss': 0.8903, 'learning_rate': 1.0888e-04, 'epoch': 1.42}
{'loss': 0.8903, 'learning_rate': 0.00010887958953229349, 'epoch': 1.42}
02/16/2024 00:52:41 - INFO - llmtuner.extras.callbacks - {'loss': 0.9117, 'learning_rate': 1.0560e-04, 'epoch': 1.45}
{'loss': 0.9117, 'learning_rate': 0.00010559530220837593, 'epoch': 1.45}
02/16/2024 00:54:29 - INFO - llmtuner.extras.callbacks - {'loss': 0.9222, 'learning_rate': 1.0230e-04, 'epoch': 1.48}
{'loss': 0.9222, 'learning_rate': 0.00010230494769863039, 'epoch': 1.48}
02/16/2024 00:56:17 - INFO - llmtuner.extras.callbacks - {'loss': 0.9100, 'learning_rate': 9.9012e-05, 'epoch': 1.51}
{'loss': 0.91, 'learning_rate': 9.901209385185345e-05, 'epoch': 1.51}
02/16/2024 00:58:05 - INFO - llmtuner.extras.callbacks - {'loss': 0.8984, 'learning_rate': 9.5720e-05, 'epoch': 1.54}
{'loss': 0.8984, 'learning_rate': 9.572031122696195e-05, 'epoch': 1.54}
02/16/2024 00:59:53 - INFO - llmtuner.extras.callbacks - {'loss': 0.9091, 'learning_rate': 9.2433e-05, 'epoch': 1.57}
{'loss': 0.9091, 'learning_rate': 9.24331692213087e-05, 'epoch': 1.57}
02/16/2024 01:01:41 - INFO - llmtuner.extras.callbacks - {'loss': 0.9352, 'learning_rate': 8.9154e-05, 'epoch': 1.60}
{'loss': 0.9352, 'learning_rate': 8.915423220025747e-05, 'epoch': 1.6}
02/16/2024 01:03:29 - INFO - llmtuner.extras.callbacks - {'loss': 0.9368, 'learning_rate': 8.5887e-05, 'epoch': 1.64}
{'loss': 0.9368, 'learning_rate': 8.588705563221444e-05, 'epoch': 1.64}
02/16/2024 01:05:18 - INFO - llmtuner.extras.callbacks - {'loss': 0.9371, 'learning_rate': 8.2635e-05, 'epoch': 1.67}
{'loss': 0.9371, 'learning_rate': 8.263518223330697e-05, 'epoch': 1.67}
02/16/2024 01:07:05 - INFO - llmtuner.extras.callbacks - {'loss': 0.9193, 'learning_rate': 7.9402e-05, 'epoch': 1.70}
{'loss': 0.9193, 'learning_rate': 7.940213812589018e-05, 'epoch': 1.7}
02/16/2024 01:08:53 - INFO - llmtuner.extras.callbacks - {'loss': 0.9103, 'learning_rate': 7.6191e-05, 'epoch': 1.73}
{'loss': 0.9103, 'learning_rate': 7.619142901504649e-05, 'epoch': 1.73}
02/16/2024 01:10:41 - INFO - llmtuner.extras.callbacks - {'loss': 0.9275, 'learning_rate': 7.3007e-05, 'epoch': 1.76}
{'loss': 0.9275, 'learning_rate': 7.300653638722463e-05, 'epoch': 1.76}
02/16/2024 01:12:29 - INFO - llmtuner.extras.callbacks - {'loss': 0.9058, 'learning_rate': 6.9851e-05, 'epoch': 1.79}
{'loss': 0.9058, 'learning_rate': 6.985091373513972e-05, 'epoch': 1.79}
02/16/2024 01:14:17 - INFO - llmtuner.extras.callbacks - {'loss': 0.8925, 'learning_rate': 6.6728e-05, 'epoch': 1.82}
{'loss': 0.8925, 'learning_rate': 6.67279828130277e-05, 'epoch': 1.82}
02/16/2024 01:16:05 - INFO - llmtuner.extras.callbacks - {'loss': 0.9008, 'learning_rate': 6.3641e-05, 'epoch': 1.86}
{'loss': 0.9008, 'learning_rate': 6.364112992631536e-05, 'epoch': 1.86}
02/16/2024 01:17:53 - INFO - llmtuner.extras.callbacks - {'loss': 0.9264, 'learning_rate': 6.0594e-05, 'epoch': 1.89}
{'loss': 0.9264, 'learning_rate': 6.0593702259728336e-05, 'epoch': 1.89}
[INFO|trainer.py:2936] 2024-02-16 01:17:53,626 >> Saving model checkpoint to saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-300
[INFO|configuration_utils.py:729] 2024-02-16 01:17:53,932 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-16 01:17:53,933 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|tokenization_utils_base.py:2433] 2024-02-16 01:17:53,986 >> tokenizer config file saved in saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-300/tokenizer_config.json
[INFO|tokenization_utils_base.py:2442] 2024-02-16 01:17:53,986 >> Special tokens file saved in saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-300/special_tokens_map.json
02/16/2024 01:19:42 - INFO - llmtuner.extras.callbacks - {'loss': 0.8975, 'learning_rate': 5.7589e-05, 'epoch': 1.92}
{'loss': 0.8975, 'learning_rate': 5.758900424781939e-05, 'epoch': 1.92}
02/16/2024 01:21:30 - INFO - llmtuner.extras.callbacks - {'loss': 0.8961, 'learning_rate': 5.4630e-05, 'epoch': 1.95}
{'loss': 0.8961, 'learning_rate': 5.463029399185217e-05, 'epoch': 1.95}
02/16/2024 01:23:17 - INFO - llmtuner.extras.callbacks - {'loss': 0.8878, 'learning_rate': 5.1721e-05, 'epoch': 1.98}
{'loss': 0.8878, 'learning_rate': 5.172077972692553e-05, 'epoch': 1.98}
02/16/2024 01:25:00 - INFO - llmtuner.extras.callbacks - {'loss': 0.8807, 'learning_rate': 4.8864e-05, 'epoch': 2.01}
{'loss': 0.8807, 'learning_rate': 4.886361634317004e-05, 'epoch': 2.01}
02/16/2024 01:26:48 - INFO - llmtuner.extras.callbacks - {'loss': 0.9260, 'learning_rate': 4.6062e-05, 'epoch': 2.04}
{'loss': 0.926, 'learning_rate': 4.6061901964787904e-05, 'epoch': 2.04}
02/16/2024 01:28:37 - INFO - llmtuner.extras.callbacks - {'loss': 0.8785, 'learning_rate': 4.3319e-05, 'epoch': 2.08}
{'loss': 0.8785, 'learning_rate': 4.3318674590646237e-05, 'epoch': 2.08}
02/16/2024 01:30:25 - INFO - llmtuner.extras.callbacks - {'loss': 0.8654, 'learning_rate': 4.0637e-05, 'epoch': 2.11}
{'loss': 0.8654, 'learning_rate': 4.063690880006671e-05, 'epoch': 2.11}
02/16/2024 01:32:13 - INFO - llmtuner.extras.callbacks - {'loss': 0.8755, 'learning_rate': 3.8020e-05, 'epoch': 2.14}
{'loss': 0.8755, 'learning_rate': 3.801951252738295e-05, 'epoch': 2.14}
02/16/2024 01:34:01 - INFO - llmtuner.extras.callbacks - {'loss': 0.9096, 'learning_rate': 3.5469e-05, 'epoch': 2.17}
{'loss': 0.9096, 'learning_rate': 3.546932390876351e-05, 'epoch': 2.17}
02/16/2024 01:35:49 - INFO - llmtuner.extras.callbacks - {'loss': 0.8935, 'learning_rate': 3.2989e-05, 'epoch': 2.20}
{'loss': 0.8935, 'learning_rate': 3.29891082047197e-05, 'epoch': 2.2}
02/16/2024 01:37:37 - INFO - llmtuner.extras.callbacks - {'loss': 0.8815, 'learning_rate': 3.0582e-05, 'epoch': 2.23}
{'loss': 0.8815, 'learning_rate': 3.058155480163493e-05, 'epoch': 2.23}
02/16/2024 01:39:25 - INFO - llmtuner.extras.callbacks - {'loss': 0.8915, 'learning_rate': 2.8249e-05, 'epoch': 2.26}
{'loss': 0.8915, 'learning_rate': 2.8249274295566864e-05, 'epoch': 2.26}
02/16/2024 01:41:13 - INFO - llmtuner.extras.callbacks - {'loss': 0.9325, 'learning_rate': 2.5995e-05, 'epoch': 2.30}
{'loss': 0.9325, 'learning_rate': 2.5994795661485437e-05, 'epoch': 2.3}
02/16/2024 01:43:00 - INFO - llmtuner.extras.callbacks - {'loss': 0.8596, 'learning_rate': 2.3821e-05, 'epoch': 2.33}
{'loss': 0.8596, 'learning_rate': 2.382056351101454e-05, 'epoch': 2.33}
02/16/2024 01:44:48 - INFO - llmtuner.extras.callbacks - {'loss': 0.9458, 'learning_rate': 2.1729e-05, 'epoch': 2.36}
{'loss': 0.9458, 'learning_rate': 2.1728935441652686e-05, 'epoch': 2.36}
02/16/2024 01:46:36 - INFO - llmtuner.extras.callbacks - {'loss': 0.9168, 'learning_rate': 1.9722e-05, 'epoch': 2.39}
{'loss': 0.9168, 'learning_rate': 1.972217948034596e-05, 'epoch': 2.39}
02/16/2024 01:48:24 - INFO - llmtuner.extras.callbacks - {'loss': 0.8889, 'learning_rate': 1.7802e-05, 'epoch': 2.42}
{'loss': 0.8889, 'learning_rate': 1.780247162418539e-05, 'epoch': 2.42}
02/16/2024 01:50:12 - INFO - llmtuner.extras.callbacks - {'loss': 0.8746, 'learning_rate': 1.5972e-05, 'epoch': 2.45}
{'loss': 0.8746, 'learning_rate': 1.5971893480895583e-05, 'epoch': 2.45}
02/16/2024 01:52:00 - INFO - llmtuner.extras.callbacks - {'loss': 0.9135, 'learning_rate': 1.4232e-05, 'epoch': 2.48}
{'loss': 0.9135, 'learning_rate': 1.423243001167337e-05, 'epoch': 2.48}
02/16/2024 01:53:48 - INFO - llmtuner.extras.callbacks - {'loss': 0.9149, 'learning_rate': 1.2586e-05, 'epoch': 2.52}
{'loss': 0.9149, 'learning_rate': 1.2585967378823448e-05, 'epoch': 2.52}
[INFO|trainer.py:2936] 2024-02-16 01:53:48,638 >> Saving model checkpoint to saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-400
[INFO|configuration_utils.py:729] 2024-02-16 01:53:48,945 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-16 01:53:48,946 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|tokenization_utils_base.py:2433] 2024-02-16 01:53:49,000 >> tokenizer config file saved in saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-400/tokenizer_config.json
[INFO|tokenization_utils_base.py:2442] 2024-02-16 01:53:49,000 >> Special tokens file saved in saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tmp-checkpoint-400/special_tokens_map.json
02/16/2024 01:55:37 - INFO - llmtuner.extras.callbacks - {'loss': 0.8727, 'learning_rate': 1.1034e-05, 'epoch': 2.55}
{'loss': 0.8727, 'learning_rate': 1.103429090052528e-05, 'epoch': 2.55}
02/16/2024 01:57:25 - INFO - llmtuner.extras.callbacks - {'loss': 0.9143, 'learning_rate': 9.5791e-06, 'epoch': 2.58}
{'loss': 0.9143, 'learning_rate': 9.57908311494896e-06, 'epoch': 2.58}
02/16/2024 01:59:13 - INFO - llmtuner.extras.callbacks - {'loss': 0.8871, 'learning_rate': 8.2219e-06, 'epoch': 2.61}
{'loss': 0.8871, 'learning_rate': 8.221921955819034e-06, 'epoch': 2.61}
02/16/2024 02:01:01 - INFO - llmtuner.extras.callbacks - {'loss': 0.8945, 'learning_rate': 6.9643e-06, 'epoch': 2.64}
{'loss': 0.8945, 'learning_rate': 6.964279041404553e-06, 'epoch': 2.64}
02/16/2024 02:02:49 - INFO - llmtuner.extras.callbacks - {'loss': 0.8655, 'learning_rate': 5.8075e-06, 'epoch': 2.67}
{'loss': 0.8655, 'learning_rate': 5.80751807879103e-06, 'epoch': 2.67}
02/16/2024 02:04:37 - INFO - llmtuner.extras.callbacks - {'loss': 0.9161, 'learning_rate': 4.7529e-06, 'epoch': 2.70}
{'loss': 0.9161, 'learning_rate': 4.752893385164103e-06, 'epoch': 2.7}
02/16/2024 02:06:25 - INFO - llmtuner.extras.callbacks - {'loss': 0.8870, 'learning_rate': 3.8015e-06, 'epoch': 2.74}
{'loss': 0.887, 'learning_rate': 3.8015485277086205e-06, 'epoch': 2.74}
02/16/2024 02:08:13 - INFO - llmtuner.extras.callbacks - {'loss': 0.8489, 'learning_rate': 2.9545e-06, 'epoch': 2.77}
{'loss': 0.8489, 'learning_rate': 2.954515083598064e-06, 'epoch': 2.77}
02/16/2024 02:10:01 - INFO - llmtuner.extras.callbacks - {'loss': 0.8596, 'learning_rate': 2.2127e-06, 'epoch': 2.80}
{'loss': 0.8596, 'learning_rate': 2.212711521418487e-06, 'epoch': 2.8}
02/16/2024 02:11:49 - INFO - llmtuner.extras.callbacks - {'loss': 0.8849, 'learning_rate': 1.5769e-06, 'epoch': 2.83}
{'loss': 0.8849, 'learning_rate': 1.576942205240317e-06, 'epoch': 2.83}
02/16/2024 02:13:37 - INFO - llmtuner.extras.callbacks - {'loss': 0.9015, 'learning_rate': 1.0479e-06, 'epoch': 2.86}
{'loss': 0.9015, 'learning_rate': 1.0478965224176906e-06, 'epoch': 2.86}
02/16/2024 02:15:25 - INFO - llmtuner.extras.callbacks - {'loss': 0.8759, 'learning_rate': 6.2615e-07, 'epoch': 2.89}
{'loss': 0.8759, 'learning_rate': 6.261481360611332e-07, 'epoch': 2.89}
02/16/2024 02:17:13 - INFO - llmtuner.extras.callbacks - {'loss': 0.8423, 'learning_rate': 3.1215e-07, 'epoch': 2.92}
{'loss': 0.8423, 'learning_rate': 3.12154362994177e-07, 'epoch': 2.92}
02/16/2024 02:19:01 - INFO - llmtuner.extras.callbacks - {'loss': 0.8736, 'learning_rate': 1.0626e-07, 'epoch': 2.96}
{'loss': 0.8736, 'learning_rate': 1.0625567786842761e-07, 'epoch': 2.96}
02/16/2024 02:20:48 - INFO - llmtuner.extras.callbacks - {'loss': 0.9235, 'learning_rate': 8.6753e-09, 'epoch': 2.99}
{'loss': 0.9235, 'learning_rate': 8.675343974762219e-09, 'epoch': 2.99}
[INFO|trainer.py:1962] 2024-02-16 02:21:27,313 >> 

Training completed. Do not forget to share your model on huggingface.co/models =)


02/16/2024 02:21:27 - INFO - llmtuner.extras.callbacks - {'loss': 0.0000, 'learning_rate': 0.0000e+00, 'epoch': 3.00}
{'train_runtime': 10294.3295, 'train_samples_per_second': 2.962, 'train_steps_per_second': 0.046, 'train_loss': 0.9314494977707133, 'epoch': 3.0}
[INFO|trainer.py:2936] 2024-02-16 02:21:27,315 >> Saving model checkpoint to saves/InternLM-7B/lora/train_2024-02-15-23-26-12
[INFO|configuration_utils.py:729] 2024-02-16 02:21:27,629 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-16 02:21:27,630 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|tokenization_utils_base.py:2433] 2024-02-16 02:21:27,682 >> tokenizer config file saved in saves/InternLM-7B/lora/train_2024-02-15-23-26-12/tokenizer_config.json
[INFO|tokenization_utils_base.py:2442] 2024-02-16 02:21:27,682 >> Special tokens file saved in saves/InternLM-7B/lora/train_2024-02-15-23-26-12/special_tokens_map.json
***** train metrics *****
  epoch                    =        3.0
  train_loss               =     0.9314
  train_runtime            = 2:51:34.32
  train_samples_per_second =      2.962
  train_steps_per_second   =      0.046
[INFO|modelcard.py:452] 2024-02-16 02:21:27,688 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}
[INFO|tokenization_utils_base.py:2027] 2024-02-16 02:22:16,429 >> loading file tokenizer.model from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer.model
[INFO|tokenization_utils_base.py:2027] 2024-02-16 02:22:16,429 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2027] 2024-02-16 02:22:16,429 >> loading file special_tokens_map.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/special_tokens_map.json
[INFO|tokenization_utils_base.py:2027] 2024-02-16 02:22:16,429 >> loading file tokenizer_config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer_config.json
[INFO|tokenization_utils_base.py:2027] 2024-02-16 02:22:16,429 >> loading file tokenizer.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer.json
[INFO|configuration_utils.py:729] 2024-02-16 02:22:16,770 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-16 02:22:16,771 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

02/16/2024 02:22:16 - INFO - llmtuner.model.patcher - Quantizing model to 4 bit.
[INFO|modeling_utils.py:3476] 2024-02-16 02:22:16,775 >> loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/pytorch_model.bin.index.json
[INFO|modeling_utils.py:1426] 2024-02-16 02:22:16,777 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
[INFO|configuration_utils.py:826] 2024-02-16 02:22:16,778 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0
}

[INFO|modeling_utils.py:3615] 2024-02-16 02:22:16,944 >> Detected 4-bit loading: activating 4-bit loading for this model
Loading checkpoint shards:   0% 0/2 [00:00<?, ?it/s]

```