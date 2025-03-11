cd nanoGPT/config


# a. TINY TEST EXPERIMENTS
python script_create_config_val.py 00A
python script_create_config_val.py 00V

# b. SMALL-SCALE EXPERIMENTS
# python script_create_config_val.py 12A
# python script_create_config_val.py 12V
# [..]

# c. SCALED COUPLED ADAM
# python script_create_config_val.py 15V --filter as3-s1
# [..]
# python script_create_config_val.py 15V --filter as3200-s1

# d. SGD
# python script_create_config_val.py 15A --filter sgd-100-s1
# [..]
# python script_create_config_val.py 15A --filter sgd-600-s1