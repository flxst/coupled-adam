import argparse
import os
from os.path import abspath, dirname, join, isdir

import sys
ROOT_DIR = abspath(dirname((__file__)))
if not ROOT_DIR in sys.path:
    sys.path.append(ROOT_DIR)
from src.isotropy import run as run_isotropy
from src.embeddings_benchmark import POST_PROCESSING_FUNCS, DATASET_PATH_MAP
from src.embeddings_benchmark import main as run_embedding_benchmarks

def main(base_directory: str, output_directory: str, target_dir: str):
    prefix = "exp"
    output_directory_full = join(abspath(dirname(ROOT_DIR)), 'nanoGPT', base_directory, output_directory)
    assert isdir(output_directory_full), f"ERROR! could not find directory {output_directory_full}"
    hf_directories = [elem for elem in os.listdir(output_directory_full) if isdir(join(output_directory_full, elem)) and elem.startswith(prefix)]
    assert len(hf_directories) == 1, f"ERROR! could not find single hf subdirectory in {output_directory_full}"
    hf_directory = hf_directories[0]

    model_path = join(output_directory_full, hf_directory)
    model_name = output_directory

    print("\n---------------------------------")
    print(f"> isotropy ({output_directory})")
    run_isotropy(
        [model_path], 
        f"{target_dir}/isotropy---{model_name}.jsonl"
    )

    print(f"> ebenchmark ({output_directory})")
    run_embedding_benchmarks(
        models_list=[model_path],
        post_processing_funcs=POST_PROCESSING_FUNCS,
        dataset_path_map=DATASET_PATH_MAP,
        out_path=f"{target_dir}/ebenchmark---{model_name}.jsonl"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='output directory to analyze, e.g. "output"')
    parser.add_argument('base_directory')
    parser.add_argument('directory')
    parser.add_argument('--target_dir', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    target_dir = join(args.base_directory, args.directory) if args.target_dir else "results"
    main(args.base_directory, args.directory, target_dir)