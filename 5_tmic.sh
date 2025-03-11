cd tmic


# a. TINY TEST EXPERIMENTS
python eval.py ../nanoGPT/output exp00A-node-125M-1k-baseline-s1 --target_dir
python eval.py ../nanoGPT/output exp00V-node-125M-1k-AVG-s1 --target_dir

# b. SMALL-SCALE EXPERIMENTS
# python eval.py ../nanoGPT/output exp12A-node-125M-50k-baseline-s1.py --target_dir
# python eval.py ../nanoGPT/output exp12V-node-125M-50k-AVG-s1.py --target_dir
# [..]

# c. SCALED COUPLED ADAM
# python eval.py ../nanoGPT/output exp15V-node-125M-200k-AVG-as3-s1 --target_dir
# [..]
# python eval.py ../nanoGPT/output exp15V-node-125M-200k-AVG-as3200-s1 --target_dir

# d. SGD
# python eval.py ../nanoGPT/output exp15A-node-125M-200k-sgd-100-s1 --target_dir
# [..]
# python eval.py ../nanoGPT/output exp15A-node-125M-200k-sgd-600-s1 --target_dir