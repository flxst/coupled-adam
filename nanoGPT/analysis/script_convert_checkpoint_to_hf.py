"""
Example: python analysis/script_convert_checkpoint_to_hf.py <path-to-pt-checkpoint> cuda:1
"""

import argparse
from os.path import abspath, dirname, join, isfile, isdir
import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoTokenizer
from typing import Tuple, Dict, Any, Optional

import sys
ROOT_DIR = abspath(dirname(dirname(__file__)))
if not ROOT_DIR in sys.path:
    sys.path.append(ROOT_DIR)

from model import GPTConfig, GPT
from analysis.script_analyze import _get_last_file_in_directory

VERBOSE = False
INPUT_TEXTS = [
    "This is a first test", 
    "Hello how are you today"
]


def load_model_pt(ckpt_path, device) -> Tuple[GPT, Dict[str, Any]]:
    """
    load pytorch GPT model from ckpt_path
    """
    model_args = {}
    #################################
    # from train.py
    #################################
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in [
        'block_size',
        'vocab_size',
        'n_layer',
        'n_head',
        'n_embd',
        'dropout',
        'bias',
    ]:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    return model, model_args



class MyConfig(PretrainedConfig):
    model_type = "nanogpt"

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        super().__init__(**kwargs)


class MyModel(PreTrainedModel):
    config_class = MyConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = GPT(config)

    def forward(self, tensor):
        logits, loss = self.model(tensor)
        return argparse.Namespace(logits=logits[1], loss=loss)

def convert_to_hf(model_pt: GPT, model_args: Dict[str, Any]):
    model_config = MyConfig(**model_args)
    model_hf = MyModel(model_config)
    model_hf.model = model_pt
    if VERBOSE:
        print("model_args", model_args)
        print("model_config", type(model_config))
        print("model_hf", type(model_hf))
    return model_hf


def main(checkpoint_path: str, device: str):
    print("\n=== LOAD PT MODEL ===")
    checkpoint_path = join(ROOT_DIR, checkpoint_path)
    if not isfile(checkpoint_path):
        if not isdir(checkpoint_path):
            raise Exception(f"ERROR! could not find checkpoint_path = {checkpoint_path}")
        else:  # find out checkpoint_path
            last_checkpoint = _get_last_file_in_directory(checkpoint_path)
            checkpoint_path = join(checkpoint_path, last_checkpoint)
    print(checkpoint_path)

    print(f"> load  {checkpoint_path}")
    model_pt, model_args = load_model_pt(checkpoint_path, device)

    print("\n=== CONVERT PT MODEL TO HF MODEL ===")
    model_hf = convert_to_hf(model_pt, model_args)

    print("\n=== TEST HF MODEL ON DATA ===")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokens = tokenizer(INPUT_TEXTS, return_tensors='pt')
    X = tokens['input_ids'][:, :-1]
    X.to(device)

    if VERBOSE:
        print(tokens['input_ids'])
        print(X)

    output = model_hf(X)
    logits = output.logits
    loss = output.loss
    if VERBOSE:
        print(logits.shape)
        print(loss)
    predictions_id = torch.argmax(logits, dim=-1)
    predictions = tokenizer.batch_decode(predictions_id)

    print("Input text incl. last word --- prediction for last word")
    for input_text, prediction in zip(INPUT_TEXTS, predictions):
        print(input_text, "---", prediction)

    if 1:
        print(f"\n=== SAVE HF model ===")
        checkpoint_path_hf = checkpoint_path.replace('.pt', '.hf')
        checkpoint_path_hf = checkpoint_path_hf.replace('=', '_')  # for lm-eval harness
        checkpoint_path_hf_full = join(ROOT_DIR, checkpoint_path_hf)
        model_hf.save_pretrained(checkpoint_path_hf_full, safe_serialization=False)
        print(f"> saved HF model at {checkpoint_path_hf}")

        print(f"\n=== SAVE HF tokenizer ===")
        tokenizer.save_pretrained(checkpoint_path_hf_full)
        print(f"> saved HF tokenizer at {checkpoint_path_hf}")

    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('checkpoint_path')
    parser.add_argument('device')
    args = parser.parse_args()
    main(args.checkpoint_path, args.device)
