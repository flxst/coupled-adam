"""This script fetches results from the sister repositories lm-evaluation-harness and tooMuchInCommon"""
import argparse
import os
import shutil
from os.path import abspath, dirname, join, isdir, isfile

import sys
ROOT_DIR = abspath(dirname(dirname(__file__)))
if not ROOT_DIR in sys.path:
    sys.path.append(ROOT_DIR)

from analysis.script_analyze import _get_directories

def _check_existence_of_source_directory(source_directory: str):
    if isdir(source_directory):
        print(f"> found source_directory = {source_directory}")
    else:
        raise Exception(f"ERROR! could not find source_directory = {source_directory}")


def sync_lm_eval(target_directory: str, lm_eval_directory: str):
    print("\n========== SYNC LM-EVAL ============")
    _check_existence_of_source_directory(lm_eval_directory)
    modalities = "mn5" in target_directory

    if modalities:
        lm_eval_directories = [elem for elem in os.listdir(lm_eval_directory) if 'mn5' in elem]
    else:
        lm_eval_directories = [elem for elem in os.listdir(lm_eval_directory) if '__' in elem and '---' in elem]

    if modalities:
        lm_eval_directories = {
            elem.split('__')[-1].split('.hf')[0]: elem
            for elem in lm_eval_directories
        }
    else:
        lm_eval_directories = {
            elem.split('__')[-1].split('---')[0]: elem
            for elem in lm_eval_directories
        }

    target_directories = _get_directories(target_directory)

    for directory in target_directories:
        directory_path = join(target_directory, directory)
        json_files = [elem for elem in os.listdir(directory_path) if elem.startswith('results') and elem.endswith('.json')]
        if len(json_files) > 0:
            print(f"> {directory}: old lm-eval results")
        elif directory in lm_eval_directories:
            results_path = join(lm_eval_directory, lm_eval_directories[directory])
            assert isdir(directory_path), f"ERROR! could not find target directory {results_path}"
            json_files = [elem for elem in os.listdir(results_path) if elem.startswith('results') and elem.endswith('.json')]
            assert len(json_files) == 1, f"ERROR! could not find single json file in {results_path}: {json_files}"

            json_path_input = join(results_path, json_files[0])
            assert isfile(json_path_input), f"ERROR! could not find file {json_path_input}"

            json_path_output = join(directory_path, json_files[0])
            shutil.copy(json_path_input, json_path_output)
            print(f"> {directory}: NEW LM-EVAL RESULTS")
        else:
            print(f"> {directory}: no lm-eval results")

def sync_tmic(target_directory: str, tmic_directory: str):
    print("\n========== SYNC TMIC ============")
    _check_existence_of_source_directory(tmic_directory)

    tmic_files_isotropy = [elem for elem in os.listdir(tmic_directory) if elem.startswith('isotropy') and elem.endswith('.jsonl')]
    tmic_files_isotropy = {
        elem.split('---')[1].split('.json')[0]: elem
        for elem in tmic_files_isotropy
    }

    tmic_files_ebenchmark = [elem for elem in os.listdir(tmic_directory) if elem.startswith('ebenchmark') and elem.endswith('.jsonl')]
    tmic_files_ebenchmark = {
        elem.split('---')[1].split('.json')[0]: elem
        for elem in tmic_files_ebenchmark
    }

    assert tmic_files_isotropy.keys() == tmic_files_ebenchmark.keys(), f"ERROR! mismatch between tmic isotropy & ebenchmark files. missing in ebenchmark: {tmic_files_isotropy.keys() - tmic_files_ebenchmark.keys()}. missing in isotropy: {tmic_files_ebenchmark.keys() - tmic_files_isotropy.keys()}"

    target_directories = _get_directories(target_directory)

    for directory in target_directories:
        directory_path = join(target_directory, directory)
        assert isdir(directory_path), f"ERROR! could not find target directory {directory_path}"
        json_files = [elem for elem in os.listdir(directory_path) if elem.startswith('ebenchmark') and elem.endswith('.jsonl')]
        if len(json_files) > 0:
            print(f"> {directory}: old tmic results")
        elif directory in tmic_files_ebenchmark:
            results_path_isotropy = join(tmic_directory, tmic_files_isotropy[directory])
            json_path_output_isotropy = join(directory_path, tmic_files_isotropy[directory])
            shutil.copy(results_path_isotropy, json_path_output_isotropy)

            results_path_ebenchmark = join(tmic_directory, tmic_files_ebenchmark[directory])
            json_path_output_ebenchmark = join(directory_path, tmic_files_ebenchmark[directory])
            shutil.copy(results_path_ebenchmark, json_path_output_ebenchmark)
            print(f"> {directory}: NEW TMIC RESULTS")
        else:
            print(f"> {directory}: no tmic results")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='output directory to analyze, e.g. "output"')
    parser.add_argument('directory')
    parser.add_argument('--skip-tmic', action=argparse.BooleanOptionalAction)  # should be skipped if tmic eval.py is used with --target_dir (step 5) 
    args = parser.parse_args()
    print(args)
    target_directory = join(ROOT_DIR, args.directory)

    LM_EVAL_DIRECTORY = join(abspath(dirname(ROOT_DIR)), 'lm-evaluation-harness/results')
    sync_lm_eval(target_directory, lm_eval_directory=LM_EVAL_DIRECTORY)

    if args.skip_tmic is None:
        TMIC_DIRECTORY = join(abspath(dirname(ROOT_DIR)), 'tmic/results')
        sync_tmic(target_directory, tmic_directory=TMIC_DIRECTORY)
