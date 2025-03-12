#
#
#

# --- seed ---
seed = 1

# --- experiment ---
wandb_log = False
wandb_project = 'coupled-adam'
wandb_run_name = 'exp35V-node-760M-200k-AVG-s1'
out_dir = 'output/exp35V-node-760M-200k-AVG-s1'
compile = True

# --- batch size ---
# 6 batch size * 1024 block size * 4 gradaccum * 4 GPUs ~ 98304
gradient_accumulation_steps = 4*4
batch_size = 6
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
optimizer_embedding = 'coupled_adam'
weight_decay = 1e-1  # general
grad_clip = 1.0  # general; clip gradients at this value, or disable if == 0.0
beta1 = 0.9  # adamw
beta2 = 0.95  # adamw
# momentum = 0  # sgd

# --- model ---
# 760M
n_layer = 24
n_head = 16
n_embd = 1536
learning_rate = 2.5e-4
min_lr = 2.5e-5

# --- hyperparameters ---
warmup_iters = 100  # not super necessary potentially
