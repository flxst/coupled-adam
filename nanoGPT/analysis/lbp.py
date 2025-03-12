
import torch
import numpy as np
from typing import Optional
from os.path import isfile


def extract_lbp(model_path):
    c = torch.load(model_path, map_location='cpu')
    prefix = '_orig_mod.target_counter' if c['config']['compile'] else 'target_counter'

    try:
        lbp = c['model'][f'{prefix}.log_bias_probability'].detach().numpy()
    except KeyError:
        lbp = torch.zeros(50304)

    return lbp


def get_log_bias_probabilities_torch(path: str, path_default = Optional[str]) -> np.array:
    """
    get log bias probabilities of shape (vocab_size).

    Args:
        path, e.g. 'data/bert-base-uncased-lbp'

    Returns:
        log_bias_probabilities, e.g. array of shape (30000) 
    """
    save_flag = 0

    if isfile(path):
        lbp = np.load(path)
    else:
        if path_default is None:
            save_flag = 1
            model_path = path.rstrip(".lbp.npy")
            lbp = extract_lbp(model_path)
        else:
            bp = np.zeros(50304)
            _bp = np.load(path_default)
            bp[:_bp.shape[0]] = _bp
            bp += 10**-10
            lbp = np.log(bp)

    if save_flag:
        np.save(path, lbp)
        print(f"..saved log bias probabilities at file {path}")
        
    return lbp