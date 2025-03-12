
import json
from os.path import isfile
import numpy as np
from analysis.embeddings import get_input_embeddings_torch


def compute_Emu(path: str) -> np.array:
    save_flag = 0

    if isfile(path):
        Emu = np.load(path)
        print(f"..loaded Emu from file {path}")
    else:
        save_flag = 1
        embeddings_path = path.replace(".Emu.npy", ".embeddings.npy")
        assert isfile(embeddings_path), f"could not find embeddings at {embeddings_path}"
        e = get_input_embeddings_torch(path=embeddings_path)
        mu = np.mean(e.matrix, axis=0)
        Emu = np.matmul(e.matrix, mu)
        print(f"..computed Emu from embeddings {embeddings_path}")

    if save_flag:
        np.save(path, Emu)
        print(f"..saved Emu at file {path}")
        
    return Emu

def compute_average_acc(path: str, results_path: str) -> np.array:
    overwrite = False
    save_flag = 0
    path_std = path.replace('acc.npy', 'acc_std.npy')

    if not overwrite and isfile(path) and isfile(path_std):
        avg_acc = np.load(path)
        print(f"..loaded lm-eval-avg-acc from file {path}")

        avg_acc_std = np.load(path_std)
        print(f"..loaded lm-eval-avg-acc-std from file {path_std}")
    else:
        save_flag = 1
        acc_list, acc_std_list = [], []
        with open(results_path, 'r') as _file:
            results = json.load(_file)
            for task in results['results'].keys():
                if task != 'lambada_standard':
                    acc = results['results'][task]['acc,none']
                    acc_std = results['results'][task]['acc_stderr,none']
                    acc_list.append(acc)
                    acc_std_list.append(acc_std)
        avg_acc = np.mean(acc_list)
        avg_acc_std = np.sqrt(np.sum([elem**2 for elem in acc_std_list])) / len(acc_std_list)
        print(f"..extracted lm-eval-avg-acc = {avg_acc:.4f} ({avg_acc_std:.4f}) from {results_path}")

    if save_flag:
        np.save(path, avg_acc)
        print(f"..saved lm-eval-avg-acc at file {path}")

        np.save(path_std, avg_acc_std)
        print(f"..saved lm-eval-avg-acc-std at file {path_std}")
        
    return avg_acc, avg_acc_std

def load_isotropy(results_path: str) -> dict[str, np.array]:
    if isfile(results_path):
        with open(results_path, 'r') as f:
            isotropy = json.load(f)
            assert len(list(isotropy.values())) == 1, f"ERROR! len(list(isotropy.values())) = {len(list(isotropy.values()))} should be 1!"
            isotropy = list(isotropy.values())[0]
        print(f"..loaded isotropy from file {results_path}")
        return isotropy

def extract_cos_sim(path: str, results_path: str) -> np.array:
    save_flag = 0

    if isfile(path):
        cos_sim = np.load(path)
        print(f"..loaded cos_sim from file {path}")
    else:
        save_flag = 1
        with open(results_path, 'r') as _file:
            results = json.load(_file)
            for task in results.keys():
                if task.endswith('+original'):
                    cos_sim = results[task]['average'][0]
        print(f"..extracted tmic-ebenchmark-original-cos-sim = {cos_sim:.4f} from {results_path}")

    if save_flag:
        np.save(path, cos_sim)
        print(f"..saved tmic-ebenchmark-original-cos-sim at file {path}")
        
    return cos_sim

def extract_dot_sim(path: str, results_path: str) -> np.array:
    save_flag = 0

    if isfile(path):
        dot_sim = np.load(path)
        print(f"..loaded dot_sim from file {path}")
    else:
        save_flag = 1
        with open(results_path, 'r') as _file:
            results = json.load(_file)
            for task in results.keys():
                if task.endswith('+original'):
                    dot_sim = results[task]['average'][1]
        print(f"..extracted tmic-ebenchmark-original-dot-sim = {dot_sim:.4f} from {results_path}")

    if save_flag:
        np.save(path, dot_sim)
        print(f"..saved tmic-ebenchmark-original-dot-sim at file {path}")
        
    return dot_sim

def compute_avg_norm(path: str) -> np.array:
    save_flag = 0

    if isfile(path):
        avg_norm = np.load(path)
        print(f"..loaded avg norms from file {path}")
    else:
        save_flag = 1
        embeddings_path = path.replace(".avgnorm.npy", ".embeddings.npy")
        assert isfile(embeddings_path), f"could not find embeddings at {embeddings_path}"
        e = get_input_embeddings_torch(path=embeddings_path)
        norms = np.linalg.norm(e.matrix, axis=1)
        avg_norm = np.mean(norms)
        print(f"..computed avg norms from embeddings {embeddings_path}")

    if save_flag:
        np.save(path, avg_norm)
        print(f"..saved avg norms at file {path}")
        
    return avg_norm


def compute_mu_norm(path: str) -> np.array:
    save_flag = 0

    if isfile(path):
        mu_norm = np.load(path)
        print(f"..loaded mu norms from file {path}")
    else:
        save_flag = 1
        embeddings_path = path.replace(".munorm.npy", ".embeddings.npy")
        assert isfile(embeddings_path), f"could not find embeddings at {embeddings_path}"
        e = get_input_embeddings_torch(path=embeddings_path)
        mu_norm = np.linalg.norm(np.mean(e.matrix, axis=0))
        print(f"..computed mu norms from embeddings {embeddings_path}")

    if save_flag:
        np.save(path, mu_norm)
        print(f"..saved mu norms at file {path}")
        
    return mu_norm

def compute_all_norms(path: str) -> np.array:
    save_flag = 0

    if isfile(path):
        all_norms = np.load(path)
        print(f"..loaded all norms from file {path}")
    else:
        save_flag = 1
        embeddings_path = path.replace(".allnorm.npy", ".embeddings.npy")
        assert isfile(embeddings_path), f"could not find embeddings at {embeddings_path}"
        e = get_input_embeddings_torch(path=embeddings_path)
        all_norms = np.linalg.norm(e.matrix, axis=1)
        print(f"..computed all norms from embeddings {embeddings_path}")

    if save_flag:
        np.save(path, all_norms)
        print(f"..saved all norms at file {path}")
        
    return all_norms

def compute_ratio_norms(path: str, mu_norm: np.array, avg_norm: np.array) -> np.array:
    save_flag = 0

    if isfile(path):
        ratio_norm = np.load(path)
        print(f"..loaded ratio norm from file {path}")
    else:
        save_flag = 1
        ratio_norm = mu_norm / avg_norm
        print(f"..computed ratio norm from mu norm and avg norm")

    if save_flag:
        np.save(path, ratio_norm)
        print(f"..saved ratio norm at file {path}")
        
    return ratio_norm