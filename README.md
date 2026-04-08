# FDPKT

This project is the PyTorch implementation for FDPKT: A Feedback-Driven Programming Knowledge Tracing Model Fusing Error Types and Pass Rates.

# Dataset

In the `data` folder, we provide the dataset structure. Place the downloaded datasets in the following directories:

* BePKT: `data/BePKT/`
* Atcoder_C: `data/Atcoder_C/`
* AIZU_Cpp: `data/AIZU_Cpp/`

Each dataset directory should contain `train.csv`, `valid.csv`, and `test.csv`.

**Dataset Descriptions:**

- **BePKT**: Collected from an online programming learning system. Features exercises across multiple programming languages (C, C++, Python and Java) with rich concept annotations. [[Download](https://drive.google.com/drive/folders/1Jt6f0MV1paGLlctJqxHtdF1Vh2mUnsoV)]

- **AtCoder_C**: Collected from AtCoder. Focuses on C language exercises from competitive programming, requiring careful algorithmic efficiency consideration. [[Source](https://github.com/IBM/Project_CodeNet)]

- **AIZU_Cpp**: Sourced from Project CodeNet. Contains learning interactions from learners practicing C++ exercises on AIZU.org, a Japanese online programming education platform. [[Source](https://github.com/IBM/Project_CodeNet)]

All datasets are randomly split into training (60%), validation (20%), and test (20%) sets at the user level to prevent information leakage.

# Data Preparation

The dataset should be placed in the corresponding directory (`data/BePKT/`, `data/Atcoder_C/`, or `data/AIZU_Cpp/`). Each directory must contain three CSV files: `train.csv`, `valid.csv`, and `test.csv`.

Each CSV file contains 7 lines per student record, organized as follows:

```
student_id
problem_seq (comma-separated)
skill_seq (comma-separated)
answer_seq (comma-separated, 0 or 1)
error_feedback_seq (comma-separated, error type index)
partial_score_seq (comma-separated, 0.0 to 1.0)
```

Example for one student:
```
0
101,102,103,104
1,1,2,2
1,0,1,1
1,2,3,1
1.0,0.5,0.8,1.0
```

Where:
- `problem_seq`: sequence of problem IDs
- `skill_seq`: sequence of concept/skill IDs
- `answer_seq`: sequence of answers (1=correct, 0=incorrect)
- `error_feedback_seq`: sequence of error type indices
- `partial_score_seq`: sequence of partial scores (0.0 to 1.0)


# Setups

## Environment Requirements

* Linux operating system
* Python 3+
* PyTorch 1.7.0+
* scikit-learn 0.21.3+
* numpy 1.19.2+
* pandas
* tqdm 4.54.1+

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository_url>
cd FDPKT
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the dataset and place it in the corresponding directory:

* BePKT: `data/BePKT/`
* AtCoder_C: `data/Atcoder_C/`
* AIZU_Cpp: `data/AIZU_Cpp/`

Each dataset directory should contain `train.csv`, `valid.csv`, and `test.csv`.

### 5. Verify Installation

To verify that all dependencies are installed correctly, you can run:

```bash
python -c "import torch; import numpy; import pandas; import sklearn; import tqdm; print('All dependencies installed successfully!')"
```

## Quick Start

Run FDPKT with default settings (trains on all three datasets: BePKT, AtCoder_C, AIZU_Cpp):

```bash
python main.py
```

# Running FDPKT

Example for using FDPKT with custom parameters:
```
python main.py --dropout 0.3 --d 128 --learning_rate 0.002 --batch_size 64
```

Explanation of parameters:

* `save_prefix`: Prefix for save directory (e.g., "experiment1")
* `model_dir`: Directory name for model files (default: 'model')
* `result_dir`: Directory name for result files (default: 'result')
* `use_response_enhancement`: Enable response enhancement module (default: True)
* `use_response_change`: Enable response change enhancement module (default: True)
* `use_diagnosis_router`: Enable diagnosis router module (default: True)
* `dropout`: Dropout rate (default: 0.4)
* `d`: Embedding dimension (default: 128)
* `learning_rate`: Learning rate (default: 0.002)
* `epochs`: Maximum number of epochs (default: 200)
* `batch_size`: Batch size (default: 80)
* `min_seq`: Minimum sequence length (default: 3)
* `max_seq`: Maximum sequence length (default: 200)
* `grad_clip`: Gradient clipping value (default: 15.0)
* `patience`: Early stopping patience (default: 15)
* `cross_val_folds`: Number of cross-validation folds (default: 5)
