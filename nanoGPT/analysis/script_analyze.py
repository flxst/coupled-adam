"""
Example: python analysis/script_analyze.py output
"""

import argparse
from typing import Dict, List, Optional, Tuple
import os
from os.path import abspath, dirname, join, isdir, isfile
import csv
import numpy as np
import scipy

import sys
ROOT_DIR = abspath(dirname(dirname(__file__)))
if not ROOT_DIR in sys.path:
    sys.path.append(ROOT_DIR)
from analysis.embeddings import get_input_embeddings_torch
from analysis.lbp import get_log_bias_probabilities_torch
from analysis.isotropy import compute_isotropy
from analysis.norms import compute_avg_norm, compute_mu_norm, compute_all_norms, compute_ratio_norms, compute_Emu, compute_average_acc, extract_cos_sim, extract_dot_sim, load_isotropy
from analysis.spectrum import compute_spectrum
from analysis.embeddings import Embeddings

USE_CENTERED_ISOTROPY = False

def get_hardcoded_loss(_directory) -> float:
    # not used
    return -1.

def _get_directories(output_directory: str) -> List[str]:
    modalities = "mn5" in output_directory
    prefix = "prod" if modalities else "exp"
    directories = [elem for elem in os.listdir(output_directory) if isdir(join(output_directory, elem)) and elem.startswith(prefix)]
    directories = sorted(directories, key=lambda x: x.split(prefix)[-1])  # .split("-")[0])
    return directories

def _get_last_file_in_directory(directory_path: str) -> Optional[str]:
    modalities = "mn5" in directory_path
    suffix = ".bin" if modalities else ".pt"
    files = [elem for elem in os.listdir(directory_path) if isfile(join(directory_path, elem)) and elem.endswith(suffix)]
    if modalities:
        files = [file for file in files if "model" in file]
        files = sorted(files, key=lambda x: int(x.split("target_steps_")[-1].split("-target_tokens")[0]))
    else:
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

def get_losses(output_directory: str) -> Dict[str, float]:
    modalities = "mn5" in output_directory
    losses = {}
    directories = _get_directories(output_directory)
    if modalities:
        for directory in directories:
            directory_path = join(output_directory, directory)
            losses[directory] = float(f"{get_hardcoded_loss(directory):.2f}")
    else:
        for directory in directories:
            directory_path = join(output_directory, directory)
            last_file = _get_last_file_in_directory(directory_path)
            loss = float(last_file.split("_valloss=")[1].split(".pt")[0])
            losses[directory] = loss
    return losses

def get_test_losses_and_perplexities(output_directory: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    modalities = "mn5" in output_directory
    directories = _get_directories(output_directory)

    test_losses = {}
    perplexities = {}
    if modalities:
        for directory in directories:
            directory_path = join(output_directory, directory)
            test_losses[directory] = get_hardcoded_loss(directory)
            perplexities[directory] = np.exp(test_losses[directory])
    else:
        for directory in directories:
            directory_path = join(output_directory, directory)
            last_file = _get_last_file_in_directory(directory_path)
            test_loss_file = join(directory_path, last_file + '.valloss-openwebtext.npy')
            if isfile(test_loss_file):
                test_losses[directory] = np.load(test_loss_file)
                perplexities[directory] = np.exp(test_losses[directory])
            else:
                print(f"> WARNING! could not find file {test_loss_file}")
    return test_losses, perplexities

def get_embeddings(output_directory: str):
    directories = _get_directories(output_directory)
    for directory in directories:
        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        if isfile(f"{directory_path}/{last_file}.embeddings.npy"):
            pass
        else:
            _ = get_input_embeddings_torch(path=f"{directory_path}/{last_file}.embeddings.npy")

def get_isotropy(output_directory: str) -> Dict[str, float]:
    directories = _get_directories(output_directory)
    isotropy = {}
    for directory in directories:
        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        iso = compute_isotropy(path=f"{directory_path}/{last_file}.isotropy.npy")
        isotropy[directory] = iso
    return isotropy

def get_norms(output_directory: str) -> Tuple[Dict[str, np.array], Dict[str, np.array], Dict[str, np.array]]:
    directories = _get_directories(output_directory)
    avg_norms = {}
    mu_norms = {}
    all_norms = {}
    ratio_norms = {}
    for directory in directories:
        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        avg_norm = compute_avg_norm(path=f"{directory_path}/{last_file}.avgnorm.npy")
        avg_norms[directory] = avg_norm
        mu_norm = compute_mu_norm(path=f"{directory_path}/{last_file}.munorm.npy")
        mu_norms[directory] = mu_norm
        all_norm = compute_all_norms(path=f"{directory_path}/{last_file}.allnorm.npy")
        all_norms[directory] = all_norm
        ratio_norm = compute_ratio_norms(path=f"{directory_path}/{last_file}.rationorm.npy", mu_norm=mu_norm, avg_norm=avg_norm)
        ratio_norms[directory] = ratio_norm
    return avg_norms, mu_norms, all_norms, ratio_norms

def get_lbp(output_directory: str) -> Dict[str, np.array]:
    directories = _get_directories(output_directory)
    lbp = {}
    for directory in directories:
        if "baseline" in directory or "constant" in directory or not "bootstrap" in directory:
            path_to_gpt2_train_probability = join(ROOT_DIR, "data", "openwebtext", "gpt2-train.bin.probability.npy")
        else:
            path_to_gpt2_train_probability = None

        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        l = get_log_bias_probabilities_torch(
            path=f"{directory_path}/{last_file}.lbp.npy", 
            path_default=path_to_gpt2_train_probability)
        lbp[directory] = l

    return lbp

def get_spectrum(output_directory: str) -> Dict[str, float]:
    directories = _get_directories(output_directory)
    spectrum = {}
    for directory in directories:
        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        spec = compute_spectrum(path=f"{directory_path}/{last_file}.spectrum.npy")
        spectrum[directory] = spec
    return spectrum

def get_Emu(output_directory: str) -> Dict[str, np.array]:
    directories = _get_directories(output_directory)
    Emu = {}
    for directory in directories:
        directory_path = join(output_directory, directory)
        last_file = _get_last_file_in_directory(directory_path)
        emu = compute_Emu(path=f"{directory_path}/{last_file}.Emu.npy")
        Emu[directory] = emu
    return Emu

def get_lm_eval_results(output_directory: str) -> Dict[str, np.array]:
    directories = _get_directories(output_directory)
    lm_eval_average_acc, lm_eval_average_acc_std = {}, {}
    for directory in directories:
        directory_path = join(output_directory, directory)
        json_files = [
            elem for elem in os.listdir(directory_path) 
            if elem.startswith('results') and elem.endswith('.json')
        ]
        if len(json_files) == 1:
            results_path = join(directory_path, json_files[0])
            if isfile(results_path):
                last_file = _get_last_file_in_directory(directory_path)
                average_acc, average_acc_std = compute_average_acc(path=f"{directory_path}/{last_file}.lm-eval-avg-acc.npy",
                                                                   results_path=results_path)
                lm_eval_average_acc[directory] = average_acc
                lm_eval_average_acc_std[directory] = average_acc_std
    return lm_eval_average_acc, lm_eval_average_acc_std

def get_tmic_results(output_directory: str) -> Dict[str, np.array]:
    directories = _get_directories(output_directory)
    tmic_isotropy = {}
    tmic_ebenchmark_original_cos_sim = {}
    tmic_ebenchmark_original_dot_sim = {}
    for directory in directories:
        directory_path = join(output_directory, directory)

        # isotropy
        json_files_isotropy = [
            elem for elem in os.listdir(directory_path) 
            if elem.startswith('isotropy') and elem.endswith('.jsonl') and directory in elem
        ]
        if len(json_files_isotropy) == 1:
            results_path = join(directory_path, json_files_isotropy[0])
            if isfile(results_path):
                last_file = _get_last_file_in_directory(directory_path)
                isotropy = load_isotropy(results_path)
                tmic_isotropy[directory] = isotropy

        # ebenchmark
        json_files_ebenchmark = [
            elem for elem in os.listdir(directory_path) 
            if elem.startswith('ebenchmark') and elem.endswith('.jsonl') and directory in elem
        ]
        if len(json_files_ebenchmark) == 1:
            results_path = join(directory_path, json_files_ebenchmark[0])
            if isfile(results_path):
                last_file = _get_last_file_in_directory(directory_path)
                cos_sim = extract_cos_sim(
                    path=f"{directory_path}/{last_file}.tmic-ebenchmark-original-cos-sim.npy",
                    results_path=results_path
                )
                tmic_ebenchmark_original_cos_sim[directory] = cos_sim
                dot_sim = extract_dot_sim(
                    path=f"{directory_path}/{last_file}.tmic-ebenchmark-original-dot-sim.npy",
                    results_path=results_path
                )
                tmic_ebenchmark_original_dot_sim[directory] = dot_sim
    return tmic_ebenchmark_original_cos_sim, tmic_ebenchmark_original_dot_sim, tmic_isotropy

def main(output_directory: str):
    get_embeddings(output_directory)
    Emu = get_Emu(output_directory)
    iso = get_isotropy(output_directory)
    avg_norms, mu_norms, all_norms, ratio_norms = get_norms(output_directory)
    lbp = get_lbp(output_directory)
    losses = get_losses(output_directory)
    test_losses, perplexities = get_test_losses_and_perplexities(output_directory)
    lm_eval_average_acc, lm_eval_average_acc_std = get_lm_eval_results(output_directory)
    tmic_ebenchmark_original_cos_sim, tmic_ebenchmark_original_dot_sim, tmic_isotropy = get_tmic_results(output_directory)
    if USE_CENTERED_ISOTROPY is True:
        iso = {k: np.array(tmic_isotropy[k]['centered']) for k in iso}
        print("-> use centered isotropy!")
    corr_norm_pearson = {
        directory: scipy.stats.pearsonr(all_norms[directory], lbp[directory]).statistic
        for directory in losses.keys()
        if directory in all_norms.keys() and directory in lbp.keys() and lbp[directory] is not None
    }
    corr_norm_spearman = {
        directory: scipy.stats.spearmanr(all_norms[directory], lbp[directory]).statistic
        for directory in losses.keys()
        if directory in all_norms.keys() and directory in lbp.keys() and lbp[directory] is not None
    }
    corr_emu_pearson = {
        directory: scipy.stats.pearsonr(Emu[directory], lbp[directory]).statistic
        for directory in losses.keys()
        if directory in all_norms.keys() and directory in lbp.keys() and lbp[directory] is not None
    }
    corr_emu_spearman = {
        directory: scipy.stats.spearmanr(Emu[directory], lbp[directory]).statistic
        for directory in losses.keys()
        if directory in all_norms.keys() and directory in lbp.keys() and lbp[directory] is not None
    }
    spec = get_spectrum(output_directory)

    # print
    print("\n=====================================================================")

    # csv file
    rows = [
        [
            'name', 'test_loss (~)', 'test_loss', 'test_perplexity', 'lm-eval-avg-acc', 'lm-eval-avg-acc-std', 'isotropy', 'avg norm', 'mu norm', 'ratio norm', 'eb_cos_sim', 'eb_dot_sim', 'corr_p(norm, lbp)', 'corr_s(norm, lbp)', 
            'corr_p(Emu, lbp)', 'corr_s(Emu, lbp)', 'Smin', 'Smax', 'Smin/Smax',
        ]
    ]
    for directory, loss in losses.items():
        _test_loss = f"{test_losses[directory]:.3f}" if directory in test_losses.keys() else ""
        _test_perplexity = f"{perplexities[directory]:.3f}" if directory in perplexities.keys() else ""
        _lm_eval_avg_acc = f"{lm_eval_average_acc[directory]:.3f}" if directory in lm_eval_average_acc.keys() else ""
        _lm_eval_avg_acc_std = f"{lm_eval_average_acc_std[directory]:.3f}" if directory in lm_eval_average_acc_std.keys() else ""
        _iso = f"{iso[directory]:.3f}" if directory in iso.keys() else ""
        _avg_norms = f"{avg_norms[directory]:.3f}" if directory in avg_norms.keys() else ""
        _mu_norms = f"{mu_norms[directory]:.3f}" if directory in mu_norms.keys() else ""
        _ratio_norms = f"{ratio_norms[directory]:.3f}" if directory in ratio_norms.keys() else ""
        _eb_cos_sim = f"{tmic_ebenchmark_original_cos_sim[directory]:.1f}" if directory in tmic_ebenchmark_original_cos_sim.keys() else ""
        _eb_dot_sim = f"{tmic_ebenchmark_original_dot_sim[directory]:.1f}" if directory in tmic_ebenchmark_original_dot_sim.keys() else ""
        _corr_norm_pearson = f"{corr_norm_pearson[directory]:.3f}" if directory in corr_norm_pearson.keys() else ""
        _corr_norm_spearman = f"{corr_norm_spearman[directory]:.3f}" if directory in corr_norm_spearman.keys() else ""
        _corr_emu_pearson = f"{corr_emu_pearson[directory]:.3f}" if directory in corr_emu_pearson.keys() else ""
        _corr_emu_spearman = f"{corr_emu_spearman[directory]:.3f}" if directory in corr_emu_spearman.keys() else ""
        if _corr_norm_pearson == "nan":
            _corr_norm_pearson = ""
        if _corr_norm_spearman == "nan":
            _corr_norm_spearman = ""
        if _corr_emu_pearson == "nan":
            _corr_emu_pearson = ""
        if _corr_emu_spearman == "nan":
            _corr_emu_spearman = ""
        _Smin = f"{np.min(spec[directory]):.2f}" if directory in spec.keys() else ""
        _Smax = f"{np.max(spec[directory]):.2f}" if directory in spec.keys() else ""
        _Sratio = f"{100*np.min(spec[directory])/np.max(spec[directory]):.2f}%" if directory in spec.keys() else ""
        row = [
            directory, loss, _test_loss, _test_perplexity, _lm_eval_avg_acc, _lm_eval_avg_acc_std, _iso, _avg_norms, _mu_norms, _ratio_norms,
            _eb_cos_sim, _eb_dot_sim, _corr_norm_pearson, _corr_norm_spearman, 
            _corr_emu_pearson, _corr_emu_spearman, _Smin, _Smax, _Sratio
        ]
        rows.append(row)

        # print
        print(
            directory.ljust(40), 
            f"{loss:.2f}", 
            _test_loss,
            _test_perplexity,
            _lm_eval_avg_acc,
            _lm_eval_avg_acc_std,
            _iso,
            _avg_norms,
            _mu_norms,
            _ratio_norms,
            _eb_cos_sim,
            _eb_dot_sim,
            _corr_norm_pearson,
            _corr_norm_spearman,
            _corr_emu_pearson,
            _corr_emu_spearman,
            _Smin,
            _Smax,
            _Sratio,
        )

    filename = join(output_directory, "loss_overview.csv")
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)
    print(f"> wrote table to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='output directory to analyze, e.g. "output"')
    parser.add_argument('directory')
    args = parser.parse_args()
    output_directory = join(ROOT_DIR, args.directory)
    main(output_directory)
