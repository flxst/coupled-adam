
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import torch


class Embeddings:
    """
    class for embeddings, which is a 2D array of shape (embedding dimension, N), where N can be
    - N = vocabulary, e.g. for input embeddings
    - N = examples, e.g. for keyword embeddings
    """
    @classmethod
    def from_saved_matrix(cls, path: str):
        if isfile(path):
            matrix = np.load(path)
        else:
            raise Exception(f"ERROR! could not find file {path}")

        return Embeddings(matrix)

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.shape = self.matrix.shape  # (embedding dimension, N)
        self.rank = None

    def save_matrix(self, path: str):
        dir_path = "/".join(path.split("/")[:-1])
        if isdir(dir_path):
            np.save(path, self.matrix)
        else:
            raise Exception(f"ERROR! could not find folder {dir_path}")


def extract_embeddings(model_path):
    c = torch.load(model_path, map_location='cpu')

    modalities = "mn5" in model_path

    if modalities:
        E = c['lm_head.weight'].detach().numpy()
        m_bias, norm_weight, norm_bias = None, None, None
    else:
        prefix = '_orig_mod.lm_head' if c['config']['compile'] else 'lm_head'
        
        try:
            m_bias = c['model'][f'{prefix}.m'].detach().numpy()
        except KeyError:
            try:
                m_bias = c['model'][f'{prefix}.multiplicative_bias'].detach().numpy()
            except KeyError:
                try:
                    m_bias = c['model'][f'{prefix}.V'].detach().numpy()
                except KeyError:
                    m_bias = None
        try:
            norm_weight = c['model'][f'{prefix}.layer_norm.weight'].detach().numpy()
        except KeyError:
            norm_weight = None
        try:
            norm_bias = c['model'][f'{prefix}.layer_norm.bias'].detach().numpy()
        except KeyError:
            try:
                norm_bias = c['model'][f'{prefix}.layer_norm_bias'].detach().numpy()
            except KeyError:
                norm_bias = None

        E = c['model'][f'{prefix}.embedding.weight'].detach().numpy()

    return E, m_bias, norm_weight, norm_bias


def get_input_embeddings_torch(path: str) -> Embeddings:
    """
    get input embedding matrix of shape (embedding dimension, vocabulary).
    if _vocab is specified, sample random columns and get matrix of shape (embedding dimension, _vocab)

    Args:
        path, e.g. 'data/bert-base-uncased'

    Returns:
        input_embeddings, e.g. np array of shape (768, 30000)
    """
    save_flag = 0

    if isfile(path):
        embeddings = Embeddings.from_saved_matrix(path).matrix
        print(f"..loaded input embeddings from file {path}")
    else:
        save_flag = 1
        model_path = path.split(".embeddings.npy")[0]
        embeddings, _, _, _ = extract_embeddings(model_path)
        print(f"..loaded input embeddings from model {model_path}")

    embeddings = Embeddings(embeddings)

    if save_flag:
        embeddings.save_matrix(path)
        print(f"..saved input embeddings at file {path}")

    return embeddings