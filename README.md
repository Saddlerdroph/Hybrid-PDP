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
- `datasets/real-world/` — real-world datasets.
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
## Dataset Acquisition and Preprocessing
Due to copyright restrictions, I am unable to provide the raw files of of real-world dataset. However, I will provide the access link as well as the preprocessing procedures. Each dataset have 3 columns named `record_id`, `user_id` and `value`.

### **CK Pay**

**Access**
- Download the dataset from: https://catalog.data.gov/dataset/employee-payroll
- Use the file **Employee_Payroll.csv**

**Preprocessing**
1. **record_id**
   - Assign each row a unique `record_id`, starting from **1** and increasing sequentially.

2. **user_id**
   - Group rows by the combination of **`Bureau`** and **`Employee identify`**.
   - Assign the same `user_id` to rows where both fields have identical values.
   - `user_id` should start from **1** and increase sequentially across unique `(Bureau, Employee identify)` pairs.

3. **value**
   - Take the **`Base pay`** field.
   - Round it to the nearest integer.
   - Truncate the result so that the minimum value is **0** (i.e., negative values become 0).

---
### **NY Salary**

**Access**
- Download the dataset from: https://catalog.data.gov/dataset/salary-information-for-state-authorities
- Use the file **Salary_Information_for_State_Authorities.csv**

**Preprocessing**
1. **record_id**
   - Assign each row a unique `record_id`, starting from **1** and increasing sequentially.

2. **user_id**
   - Group rows by the combination of the following four fields:
     - **`Authority Name`**
     - **`Last Name`**
     - **`Middle Initial`**
     - **`First Name`**
   - Assign the same `user_id` to rows where all four fields have identical values.
   - `user_id` should start from **1** and increase sequentially across unique combinations of these fields.

3. **value**
   - Take the **salary** field.
   - Round it to the nearest integer.
   - Truncate the result so that the minimum value is **0** (i.e., negative values become 0).

---

### **VLE**

**Access**
- Download the dataset from: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- Use the file **studentVle.csv**

**Preprocessing**
1. **record_id**
   - Assign each row a unique `record_id`, starting from **1** and increasing sequentially.

2. **user_id**
   - Use the **`id_student`** field to define users.
   - Reassign `user_id` starting from **1**, increasing sequentially across unique `id_student` values.
   - Rows with the same `id_student` must share the same `user_id`.

3. **value**
   - Take the **`sum_click`** field as `value`.
   - Round it to the nearest integer.
   - Truncate the result so that the minimum value is **0** (i.e., negative values become 0).

---

### **Online Retail**

**Access**
- Download the dataset from: https://www.kaggle.com/datasets/tunguz/online-retail
- Use the file **Online_Retail.csv**

**Preprocessing**
1. **record_id**
   - Assign each row a unique `record_id`, starting from **1** and increasing sequentially.

2. **user_id**
   - Use the **`InvoiceNo`** field to define users.
   - Reassign `user_id` starting from **1**, increasing sequentially across unique `InvoiceNo` values.
   - Rows with the same `InvoiceNo` must share the same `user_id`.

3. **value**
   - Compute `value` as:
     ```
     Quantity × UnitPrice
     ```
   - Round the result to the nearest integer.
   - Truncate the result so that the minimum value is **0** (i.e., negative values become 0).

---

### **Czech Bank**

**Access**
- Download the dataset from: https://www.kaggle.com/datasets/marceloventura/the-berka-dataset
- Use the file **trans_comma.csv**

**Preprocessing**
1. **record_id**
   - Assign each row a unique `record_id`, starting from **1** and increasing sequentially.

2. **user_id**
   - Use the **`account_id`** field to define users.
   - Reassign `user_id` starting from **1**, increasing sequentially across unique `account_id` values.
   - Rows with the same `account_id` must share the same `user_id`.

3. **value**
   - Take the **`amount`** field as `value`.
   - Round it to the nearest integer.
   - Truncate the result so that the minimum value is **0** (i.e., negative values become 0).

---

## Reprodue the Experiments
This script generates a synthetic dataset with configurable scale parameters: record values are sampled from a Gaussian distribution, and the number of records per user is sampled from a Poisson distribution.


### 1.Generating synthetic datasets

Run:
```bash
python dataset_generator.py
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
- **`-tau, --tau`**: The **τ (tau)** parameter required by `DB.db(...)` (required).  

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
