CUDA=4,5,6,7
cd nanoGPT

# a. TINY TEST EXPERIMENTS
CUDA_VISIBLE_DEVICES=$CUDA torchrun --standalone --nproc_per_node=4 train.py config/exp00A-node-125M-1k-baseline-s1.py
CUDA_VISIBLE_DEVICES=$CUDA torchrun --standalone --nproc_per_node=4 train.py config/exp00V-node-125M-1k-AVG-s1.py

# b. SMALL-SCALE EXPERIMENTS
# CUDA_VISIBLE_DEVICES=$CUDA torchrun --standalone --nproc_per_node=4 train.py config/exp12A-node-125M-50k-baseline-s1.py
# CUDA_VISIBLE_DEVICES=$CUDA torchrun --standalone --nproc_per_node=4 train.py config/exp12V-node-125M-50k-AVG-s1.py
# [..]

# c. SCALED COUPLED ADAM
# CUDA_VISIBLE_DEVICES=$CUDA torchrun --standalone --nproc_per_node=4 train.py config/exp15V-node-125M-200k-AVG-as3-s1
# [..]
# CUDA_VISIBLE_DEVICES=$CUDA torchrun --standalone --nproc_per_node=4 train.py config/exp15V-node-125M-200k-AVG-as3200-s1

# d. SGD
# CUDA_VISIBLE_DEVICES=$CUDA torchrun --standalone --nproc_per_node=4 train.py config/exp15A-node-125M-200k-sgd-100-s1
# [..]
# CUDA_VISIBLE_DEVICES=$CUDA torchrun --standalone --nproc_per_node=4 train.py config/exp15A-node-125M-200k-sgd-600-s1
