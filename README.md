# Hybrid-PDP: Simultaneously Enforcing both Record-level and User-level Personalized Differential Privacy

## Overview
This repository (or folder) contains the experimental setup and official implementation of our proposed framework under **Hybrid-PDP**. It include modules for count, summation and  maximum estimation, as well as supporting utilities for error analysis.

---

## File List 

- `Framework/` — Main implementation of the proposed Hybrid-PDP framework.
- `Doubly/` — Implementation of the doubly-truncation baseline.
- `Prefix/` — Implementation of the prefix-truncation baseline.
- `Naive/` — Implementation of the naive global-bound baseline.
- `downstream/` — Downstream DP mechanisms for Count/Sum/Max queries.
- `datasets/` — All datasets used in experiments (real-world and synthetic).
- `datasets/real-world/` — Real-world datasets and their processed versions.
- `datasets/Synthetic/` — Synthetic datasets.
- `dataset_generator.py` — Script for generating synthetic datasets.
- `privacy_generator.py` — Script for generating privacy specifications for datasets.


## Setup
To reproduce all results, please create a Python environment and install the dependencies specified in `requirements.txt`.

---
**Example (using Conda):**
```bash
conda create -n hpdp python=3.13
conda activate hpdp
pip install -r requirements.txt
```
---

## Reprodue the Experiments
This script generates a synthetic dataset with configurable scale parameters: record values are sampled from a Gaussian distribution, and the number of records per user is sampled from a Poisson distribution.


### 1.Generating synthetic datasets

Run:
```bash
python privacy_generator.py
```
#### Description of Main Parameters

- **`--n`** (required): Total number of records (rows) to generate.

- **`--out`**: Output CSV path (default: `data.csv`). The generated file contains rows in the format: `record_id,user_id,value`.

- **`--mean`**: Mean of the Gaussian distribution used to generate `value` (default: `2^20`).

- **`--std`**: Standard deviation of the Gaussian distribution used to generate `value` (**must be > 0**) (default: `2^20`).

- **`--check`**: Lower bound for `value` (inclusive). Values sampled from the Gaussian distribution are re-sampled until they fall within the range `[check, hat]` (default: `0.0`).

- **`--hat`**: Upper bound for `value` (inclusive). Values sampled from the Gaussian distribution are re-sampled until they fall within the range `[check, hat]` (default: `2^30`).

- **`--lamda`**: Mean (λ) of the Poisson distribution controlling how many records each user has (**must be > 0**) (default: `16.0`).

- **`--min_count`**: Minimum number of records per user (inclusive). If a Poisson draw is below this value, it is re-sampled (default: `1`).

- **`--max_count`**: Maximum number of records per user (inclusive). If a Poisson draw exceeds this value, it is clamped down to `max_count` (default: `2^16`).

- **`--seed`**: Random seed for reproducibility (default: `None`, meaning non-deterministic).

- **`--no_header`**: Compatibility flag (currently unused). When provided, it does not change output behavior (default: not set).


**Example Usage:**
```bash
python dataset_generator.py --n 100000 --out data.csv --lamda 8
```
---



### 2.Generating privacy specification

Run:
```bash
python privacy_generator.py
```
This script generates **synthetic privacy specifications** for a given dataset by sampling Gaussian noise and converting it into values via `2^noise`. It supports three specification types — `normal`, `linear`, and `sqrt`.

#### Description of Main Parameters

- **`-i, --in`** (infile, required): Path to the input CSV file.

- **`-o, --out`**: Output CSV path (default: output.csv).

- **`--mean`**: Mean of Gaussian noise (default: 0.0).

- **`--std`**: Standard deviation of Gaussian noise (must be >= 0) (default: 0.5).

- **`--seed`**: Random seed for repro---
python3 gen_data.py --n 100000 --out synthetic.csvducibility (default: None, meaning non-deterministic).

- **`--check`**: Exponent lower bound check used in clipping; lower clip value is 2^check (default: -20).

- **`-hat`** Exponent upper bound hat used in clipping; upper clip value is 2^hat (default: 4).

- **`--type`**: Post-transform mode — choose from normal, linear, sqrt (default: normal).

- **`--scale`**: Extra exponent scaling factor used by linear/sqrt types via 2^scale (default: 0.0).

**Example Usage:**
```bash
python privacy_generator.py -i ./data.csv --type sqrt --scale -10 
```
---




### 3. Running our framework

Run:
```bash
python Framework/run.py
```
This script integrates and invokes `ISHM.py`: the algorithm of the general framework proposed in our paper to execute three optional queries (*count*, *sum*, and *max*) under Hybrid-PDP. It also provides functionality to collect, summarize, and report utility and efficiency statistics of the algorithm's execution results.
#### Description of Main Parameters
- **`dataset_file`**: Path to the input dataset file. Each non-empty, non-comment line should contain `record_id user_id record_value`.  
- **`-r, --rounds`**: Number of times to run `ISHM.py` for repeated trials and statistical summarization (default: `1`).  
- **`-beta, --beta`**: Failure probability parameter β used by algorithm (must satisfy `0 < beta < 1`) (default: `0.1`).  
- **`-q, --query`**: Downstream query type to execute — choose from `count`, `sum`, or `max` (default: `count`).  


**Example Usage:**
```bash
python Framework/run.py datasets/real-world/preprocessed/Guassian/University/data.csv -r 50 -beta 0.05 -q sum
```
---

### 4. Running naive global-bound baseline

Run:
```bash
python Naive/run.py
```
This script integrates and invokes `strawman.py`: the algorithm of the strawman baseline in our paper to execute three optional queries (*count*, *sum*, and *max*) under Hybrid-PDP. It also provides functionality to collect, summarize, and report utility and efficiency statistics of the algorithm's execution results.
#### Description of Main Parameters
- **`dataset_file`**: Path to the input dataset file. Each non-empty, non-comment line should contain `record_id user_id record_value`.  
- **`-r, --rounds`**: Number of times to run `strawman.py` for repeated trials and statistical summarization (default: `1`).  
- **`-beta, --beta`**: Failure probability parameter β used by algorithm (must satisfy `0 < beta < 1`) (default: `0.1`).  
- **`-q, --query`**: Downstream query type to execute — choose from `count`, `sum`, or `max` (default: `count`).  


**Example Usage:**
```bash
python Naive/run.py datasets/real-world/preprocessed/Guassian/University/data.csv -r 50 -beta 0.05 -q sum
```
---


### 5. Running prefix-truncation baseline

Run:
```bash
python Prefix/run.py
```
This script integrates and invokes `PT.py`: the algorithm of *prefix-truncation* proposed in our paper to execute three optional queries (*count*, *sum*, and *max*) under Hybrid-PDP. It also provides functionality to collect, summarize, and report utility and efficiency statistics of the algorithm's execution results.
#### Description of Main Parameters
- **`dataset_file`**: Path to the input dataset file. Each non-empty, non-comment line should contain `record_id user_id record_value`.  
- **`-r, --rounds`**: Number of times to run  `PT.py` for repeated trials and statistical summarization (default: `1`).  
- **`-beta, --beta`**: Failure probability parameter β used by algorithm (must satisfy `0 < beta < 1`) (default: `0.1`).  
- **`-q, --query`**: Downstream query type to execute — choose from `count`, `sum`, or `max` (default: `count`).  


**Example Usage:**
```bash
python Prefix/run.py datasets/real-world/preprocessed/Guassian/University/data.csv -r 50 -beta 0.05 -q sum
```
---
### 6. Running doubly-truncation baseline

Run:
```bash
python Doubly/run.py
```
This script integrates and invokes `DT.py`: the algorithm of *doubly-truncation* proposed in our paper to execute three optional queries (*count*, *sum*, and *max*) under Hybrid-PDP. It also provides functionality to collect, summarize, and report utility and efficiency statistics of the algorithm's execution results.
#### Description of Main Parameters
- **`dataset_file`**: Path to the input dataset file. Each non-empty, non-comment line should contain `record_id user_id record_value`.  
- **`-r, --rounds`**: Number of times to run  `DT.py` for repeated trials and statistical summarization (default: `1`).  
- **`-beta, --beta`**: Failure probability parameter β used by algorithm (must satisfy `0 < beta < 1`) (default: `0.1`).  
- **`-q, --query`**: Downstream query type to execute — choose from `count`, `sum`, or `max` (default: `count`).  
- **`-tau, --tau`**: The **τ (tau)** parameter required by `DB.ishm(...)` (required).  

**Example Usage:**
```bash
python Doubly/run.py datasets/real-world/preprocessed/Guassian/University/data.csv -r 50 -beta 0.05 -q sum
```
---



> **Note:**  
> **Dataset-side required files (placed in the same directory as `dataset_file`):**  
> - **`record.csv`**: A mapping file with two columns: `record_id record_privacy_budget`.  
> - **`user.csv`**: A mapping file with two columns: `user_id user_privacy_budget`.  
>  
> **Execution notes:**  
> - For **`sum`** and **`max`**, `record_value` must be numeric (parseable as `float`); otherwise the script will raise an error.  
> - The script prints per-round results (including runtime and error metrics) and writes a summary row to `<dataset_dir>/{count|sum|max}.csv`.  
>  
> **Implementation notes:**  
> - Our downstream implementations for the **`sum`** and **`max`** queries follow the algorithmic descriptions in **[1]**.  
> - Our implementation of the **prefix-truncation** baseline follows the algorithmic descriptions in **[2]**.  
> - Since real-world datasets come in different formats and the preprocessing pipelines are not unified (with non-trivial requirements), please refer to the **Appendix** of our paper for the detailed preprocessing procedure.  
>
> **Reference:**  
> [1] Wei Dong and Ke Yi. *Universal Private Estimators.* In *Proceedings of the 42nd ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems (PODS)*, pages 195–206, 2023.  
> [2] Xinghe Chen, Dajun Sun, Quanqing Xu, and Wei Dong. *A General Framework for Per-record Differential Privacy.* *Proceedings of the ACM on Management of Data (PACMMOD)*, 3(6):1–27, 2025.  
