CUDA=7
cd lm-evaluation-harness

# a. TINY TEST EXPERIMENTS
python evaluate.py ../nanoGPT/output/exp00A-node-125M-1k-baseline-s1 $CUDA
python evaluate.py ../nanoGPT/output/exp00V-node-125M-1k-AVG-s1 $CUDA

# b. SMALL-SCALE EXPERIMENTS
# python evaluate.py ../nanoGPT/output/exp12A-node-125M-50k-baseline-s1.py $CUDA
# python evaluate.py ../nanoGPT/output/exp12V-node-125M-50k-AVG-s1.py $CUDA
# [..]

# c. SCALED COUPLED ADAM
# python evaluate.py ../nanoGPT/output/exp15V-node-125M-200k-AVG-as3-s1 $CUDA
# [..]
# python evaluate.py ../nanoGPT/output/exp15V-node-125M-200k-AVG-as3200-s1 $CUDA

# d. SGD
# python evaluate.py ../nanoGPT/output/exp15A-node-125M-200k-sgd-100-s1 $CUDA
# [..]
# python evaluate.py ../nanoGPT/output/exp15A-node-125M-200k-sgd-600-s1 $CUDA