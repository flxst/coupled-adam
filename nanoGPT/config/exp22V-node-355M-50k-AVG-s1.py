#
#
#

# --- seed ---
seed = 1

# --- experiment ---
wandb_log = False
wandb_project = 'coupled-adam'
wandb_run_name = 'exp22V-node-355M-50k-AVG-s1'
out_dir = 'output/exp22V-node-355M-50k-AVG-s1'
compile = True

# --- batch size ---
# 12 batch size * 1024 block size * 2 gradaccum * 4 GPUs ~ 98304
gradient_accumulation_steps = 2*4
batch_size = 12
block_size = 1024

# --- dataset size ---
# tokens ~ 98304 * 50000 ~ 4.9B
max_iters = 50000
lr_decay_iters = 50000

# --- checkpointing ---
eval_interval = 5000
eval_iters = 100
log_interval = 1000

# --- optimizer ---
optimizer_core = 'adamw'
optimizer_embedding = 'coupled_adamw'
weight_decay = 1e-1  # general
grad_clip = 1.0  # general; clip gradients at this value, or disable if == 0.0
beta1 = 0.9  # adamw
beta2 = 0.95  # adamw
# momentum = 0  # sgd

# --- model ---
# 355M
n_layer = 24
n_head = 16
n_embd = 1024
learning_rate = 3e-4
min_lr = 3e-5

# --- hyperparameters ---
warmup_iters = 100  # not super necessary potentially
