
import argparse
import subprocess
import os
from os.path import isfile, isdir, join, abspath, dirname
from typing import Optional

import sys
ROOT_DIR = abspath(dirname(dirname(__file__)))  # common root dir of lm-evaluation-harness & nanogpt
if not ROOT_DIR in sys.path:
    sys.path.append(ROOT_DIR)


def _get_last_file_in_directory(directory_path: str) -> Optional[str]:
    files = [elem for elem in os.listdir(directory_path) if isfile(join(directory_path, elem)) and elem.endswith(".pt")]
    files = sorted(files, key=lambda x: int(x.split("ckpt_step=")[-1].split("_valloss")[0]))
    if len(files) == 0:
        files = [elem for elem in os.listdir(directory_path) if isfile(join(directory_path, elem))]
        try:
            last_file = [elem for elem in files if elem.endswith('embeddings.npy')][0]
        except IndexError:
            raise Exception(f"no files found in {directory_path}")
        last_file = last_file.split(".embeddings.npy")[0]
        return last_file
    try:
        last_file = files[-1]
    except IndexError:
        files = [elem for elem in os.listdir(directory_path) if isfile(join(directory_path, elem)) and elem.endswith(".pt.embeddings.npy")]
        last_file = files[0].rstrip(".embeddings.npy")
    return last_file


def main(checkpoint_path: str, device: str, tasks: str):
    modalities = "mn5" in checkpoint_path
    suffix = ".bin" if modalities else ".pt"

    print("\n=== FIND HF MODEL ===")
    checkpoint_path = join(ROOT_DIR, 'nanoGPT', checkpoint_path)
    if not isfile(checkpoint_path):
        if not isdir(checkpoint_path):
            raise Exception(f"ERROR! could not find checkpoint_path = {checkpoint_path}")
        else:  # find out checkpoint_path
            if modalities:
                hf_directories = [elem for elem in os.listdir(checkpoint_path) if isdir(join(checkpoint_path, elem)) and elem.endswith('.hf')]
                assert len(hf_directories) == 1, f"ERROR! could not find single hf subdirectory in {checkpoint_path}"
                checkpoint_path_hf = join(checkpoint_path, hf_directories[0])
            else:
                last_checkpoint = _get_last_file_in_directory(checkpoint_path)
                checkpoint_path = join(checkpoint_path, last_checkpoint)
                checkpoint_path_hf = checkpoint_path.replace(suffix, '.hf')
                checkpoint_path_hf = checkpoint_path_hf.replace('=', '_')  # for lm-evaluation-harness
    assert isdir(checkpoint_path_hf), f"ERROR! could not find model at {checkpoint_path_hf}"
    print(f"> found model at {checkpoint_path}")

    print("\n=== RUN EVALUATION ===")

    command = f"lm-eval --model hf --model_args pretrained={checkpoint_path_hf} --tasks {tasks} --device cuda:{device} --batch_size auto --output_path results"
    print(command)
    subprocess.run(command, shell=True, capture_output=False, text=True)
    print(f"\n=== DONE ===")


if __name__ == "__main__":
    """
    needed methods for different benchmarks:
    'hellaswag'      - loglikelihood
    'winogrande'     - loglikelihood
    'arc_easy'       - loglikelihood
    'arc_challenge'  - loglikelihood
    'triviaqa'       - generate_until
    'truthfulqa_mc2' - loglikelihood
    'lambada'        - loglikelihood
    'race'           - loglikelihood
    """
    TASKS = 'hellaswag,winogrande,arc_easy,arc_challenge,truthfulqa_mc2,lambada,race'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('checkpoint_path')
    parser.add_argument('device')
    args = parser.parse_args()
    main(args.checkpoint_path, args.device, TASKS)