CUDA=7
cd nanoGPT

# a. TINY TEST EXPERIMENTS
python analysis/script_convert_checkpoint_to_hf.py output/exp00A-node-125M-1k-baseline-s1 cuda:$CUDA
python analysis/script_convert_checkpoint_to_hf.py output/exp00V-node-125M-1k-AVG-s1 cuda:$CUDA

# b. SMALL-SCALE EXPERIMENTS
# python analysis/script_convert_checkpoint_to_hf.py output/exp12A-node-125M-50k-baseline-s1.py cuda:$CUDA
# python analysis/script_convert_checkpoint_to_hf.py output/exp12V-node-125M-50k-AVG-s1.py cuda:$CUDA
# [..]

# c. SCALED COUPLED ADAM
# python analysis/script_convert_checkpoint_to_hf.py output/exp15V-node-125M-200k-AVG-as3-s1 cuda:$CUDA
# [..]
# python analysis/script_convert_checkpoint_to_hf.py output/exp15V-node-125M-200k-AVG-as3200-s1 cuda:$CUDA

# d. SGD
# python analysis/script_convert_checkpoint_to_hf.py output/exp15A-node-125M-200k-sgd-100-s1 cuda:$CUDA
# [..]
# python analysis/script_convert_checkpoint_to_hf.py output/exp15A-node-125M-200k-sgd-600-s1 cuda:$CUDA