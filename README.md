# Better Embeddings with Coupled Adam

This repository contains code to reproduce results from the paper https://arxiv.org/abs/2502.08441. 

TL;DR: The implementation of Coupled Adam can be found [here](https://github.com/flxst/coupled-adam/blob/ab896310a3f5642e7fb8904590207c2c38960cae/nanoGPT/coupled_adam.py#L621).

## 1. Overview

Our code is based on 3 existing open-source repositories. For each of them, we started off from a specific commit in the original repository ("Original Commit"). We then added our changes in a single commit per repository ("Our commit"). The following table gives an overview:

| Repository  | Purpose | Original Commit | Our Commit |
| -------------------------------- | ---- | ---- | ------------ |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Experiments & Upstream Evaluation | [325be85](https://github.com/karpathy/nanoGPT/commit/325be85d9be8c81b436728a420e85796c57dba7e) | [ba54c74](https://github.com/flxst/coupled-adam/commit/ba54c74ed0f197f1d2bc4f3bf54f12a33f030724) |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Downstream Evaluation | [7a1614e](https://github.com/EleutherAI/lm-evaluation-harness/commit/7a1614eb90d29b2983ffa027a7974b7ef53fba19) | [9c265c2](https://github.com/flxst/coupled-adam/commit/9c265c21878942574097669bb328ccd9ce39c4a3) |
| [tooMuchInCommon](https://github.com/danielbis) | Embedding Quality Evaluation  | [6598573](https://github.com/danielbis/tooMuchInCommon/commit/659857383f391816d4ee2e519b37420d63d83b36) | [5004594](https://github.com/flxst/coupled-adam/commit/50045948b579494fb383e99bcd169eb56ce9d071) |

The links in the "Our commit" column allow to conveniently inspect the changes with respect to the original code.

## 2. Preparation

### Create and Activate a Virtual Environment

```
# e.g. using conda
conda create -n coupled-adam python=3.11
conda activate coupled-adam
```

### Install Dependencies

```
# general
pip install torch==2.4.1 numpy transformers datasets tiktoken wandb tqdm scipy matplotlib seaborn

# lm-evaluation-harness
pip install -e lm-evaluation-harness
```


### Download OpenWebText

```
cd nanogpt
python data/openwebtext/prepare.py
```

## 3. Experiment Configs

Different example experiment configs are provided in the directory `nanoGPT/config`.

Note: 
- W&B logging is turned off by default. To turn it on, change `wandb_log = False` to `wandb_log = True` and log in to W&B. 
- The listed experiment configs only correspond to a single run with `seed = 1`. For all but the large-scale experiments, additional experiments with `seed = 2,3` were conducted.

### a. Tiny Test Experiments 

| Config Name                       | D     | N    | Adam            |
| --------------------------------- | ----- | ---- | --------------- |
| exp00A-node-125M-1k-baseline-s1   | 100M  | 125M | Standard        |
| exp00V-node-125M-1k-AVG-s1        | 100M  | 125M | Coupled         |

### b. Small-scale Experiments (Sec. 4.1 & 5.1): 

| Config Name                       | D     | N    | Adam            |
| --------------------------------- | ----- | ---- | --------------- |
| exp12A-node-125M-50k-baseline-s1  |   5B  | 125M | Standard        |
| exp12V-node-125M-50k-AVG-s1       |   5B  | 125M | Coupled         |
| exp13A-node-125M-100k-baseline-s1 |  10B  | 125M | Standard        |
| exp13V-node-125M-100k-AVG-s1      |  10B  | 125M | Coupled         |
| exp15A-node-125M-200k-baseline-s1 |  20B  | 125M | Standard        |
| exp15V-node-125M-200k-AVG-s1      |  20B  | 125M | Coupled         |

The corresponding config files for N=355M and N=760M are named `exp2??-node-355M-*k-*-s1` and `exp3??-node-760M-*k-AVG-s1`, respectively.

### c. Scaled Coupled Adam (Sec. 6.1): 

| Config Name                         | D      | N    | n  |
| ----------------------------------- | ------ | ---- | -- |
| exp15V-node-125M-200k-AVG-as3-s1    |  20B   | 125M | 5  |
| exp15V-node-125M-200k-AVG-as6-s1    |  20B   | 125M | 4  |
| exp15V-node-125M-200k-AVG-as13-s1   |  20B   | 125M | 3  |
| exp15V-node-125M-200k-AVG-as25-s1   |  20B   | 125M | 2  |
| exp15V-node-125M-200k-AVG-as50-s1   |  20B   | 125M | 1  |
| exp15V-node-125M-200k-AVG-as200-s1  |  20B   | 125M | -1 |
| exp15V-node-125M-200k-AVG-as400-s1  |  20B   | 125M | -2 |
| exp15V-node-125M-200k-AVG-as800-s1  |  20B   | 125M | -3 |
| exp15V-node-125M-200k-AVG-as1600-s1 |  20B   | 125M | -4 |
| exp15V-node-125M-200k-AVG-as3200-s1 |  20B   | 125M | -5 |


### d. SGD (Sec. 6.2): 

| Config Name                       | D     | N    | f   |
| --------------------------------- | ----- | ---- | --- |
| exp15A-node-125M-200k-sgd-100-s1  |  20B  | 125M | 100 |
| exp15A-node-125M-200k-sgd-200-s1  |  20B  | 125M | 200 |
| exp15A-node-125M-200k-sgd-300-s1  |  20B  | 125M | 300 |
| exp15A-node-125M-200k-sgd-400-s1  |  20B  | 125M | 400 |
| exp15A-node-125M-200k-sgd-500-s1  |  20B  | 125M | 500 |
| exp15A-node-125M-200k-sgd-600-s1  |  20B  | 125M | 600 |


## 4. Run Experiments & Analyze Results

The experiments can be run and analyzed step by step with the scripts listed in the following table.

| Script Name             | Purpose                                        |
| ------------------------| ---------------------------------------------- |
| 0_runs.sh               | Run training                                   |
| 1_prepare_validation.sh | Create validation config file                  |
| 2_run_validation.sh     | Run validation                                 |
| 3_convert.sh            | Convert checkpoint from PyTorch to HuggingFace |
| 4_lmeval.sh             | Run evaluation with lm-evaluation-harness      |
| 5_tmic.sh               | Run evaluation with tooMuchInCommon            |
| 6_aggregate.sh          | Aggregate all results                          |

Note:
- In each bash script,
    - the commands for some example experiments are listed
    - all but the two commands for the tiny test experiments are commented out
    - the GPUs can be specified in the first row, if applicable

- The output checkpoints from each experiments and the individual results from the analysis can be found in the subfolders of `nanogpt/output`.

- The aggregated results can be found at `nanogpt/output/loss_overview.csv`
