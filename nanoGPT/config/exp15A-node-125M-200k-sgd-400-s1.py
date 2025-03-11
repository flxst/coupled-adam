#
#
#

# --- seed ---
seed = 1

# --- experiment ---
wandb_log = False
wandb_project = 'coupled-adam'
wandb_run_name = 'exp15A-node-125M-200k-sgd-400-s1'
out_dir = 'output_as/exp15A-node-125M-200k-sgd-400-s1'
compile = True

# --- batch size ---
# 12 batch size * 1024 block size * 2 gradaccum * 4 GPUs ~ 98304
gradient_accumulation_steps = 2*4
batch_size = 12
block_size = 1024

# --- dataset size ---
# tokens ~ 98304 * 200000 ~ 19.7B
max_iters = 200000
lr_decay_iters = 200000

# --- checkpointing ---
eval_interval = 20000
eval_iters = 100
log_interval = 1000

# --- optimizer ---
optimizer_core = 'adamw'
optimizer_embedding = 'sgd'
weight_decay = 1e-1  # general
grad_clip = 1.0  # general; clip gradients at this value, or disable if == 0.0
beta1 = 0.9  # adamw
beta2 = 0.95  # adamw
momentum = 0.9  # sgd
lr_sgd_factor = 400

# --- model ---
# 125M
n_layer = 12
n_head = 12
n_embd = 768

# --- hyperparameters ---
warmup_iters = 100  # not super necessary potentially
