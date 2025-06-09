# Repository for Research Paper "Instance Segmentation Model Retrieval using Performance Prediction and Learning to Rank"

## Project Directory Structure

```
ismr/
├── data/                                   # Data root directory
│   ├── train/                              # Training data folder
│   │   ├── train.csv                       # Info about train data (metadata, AP scores, etc.)
│   │   ├── <category1>/                    # Category folder (e.g., basil-seed, beads, etc.)
│   │   │   ├── images/                     # Images for this category
│   │   │   └── annotations.json            # Manual instance segmentation annotation for this category
│   │   ├── <category2>/                    # Another category folder
│   │   │   ├── images/
│   │   │   └── annotations.json
│   │   └── ... (more categories)
│   ├── test/                               # Test data folder
│   │   ├── test.csv                        # Info about test data (metadata, AP scores, etc.)
│   │   ├── <category1>/                    # Category folder (e.g., almond, button, etc.)
│   │   │   ├── images/                     # Images for this category
│   │   │   └── annotations.json            # Manual instance segmentation annotation for this category
│   │   ├── <category2>/
│   │   │   ├── images/
│   │   │   └── annotations.json
│   │   └── ... (more categories)
├── ismr/                                   # Source code package
│   ├── __init__.py
│   ├── utils.py                            # Utility functions
│   ├── baselines.py                        # Baselines
│   ├── dataset.py                          # Dataset utils
│   ├── evaluator.py                        # Evaluation
│   ├── models.py                           # Proposed method model definition
├── README.md                               # Project documentation
├── pyproject.toml                          # Python project metadata (for uv)
├── .gitignore                              # Git ignore rules
```

## How to install

### Clone repository

```bash
git clone --recursive https://github.com/ohshimalab/ismr.git
```

### Install uv

We use uv as python package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install libraries

```bash
uv sync
```

## How to run codes

### Train and save proposed method

```bash
uv run python main.py train-proposed-method \
  --seed 0 \
  --input-size 224 \
  --train-csv ./data/train/train.csv \
  --train-data-dir ./data/train \
  --batch-size 32 \
  --num-workers 4 \
  --max-epochs 100 \
  --lr 0.001 \
  --alpha 0.5 \
  --neural-ndcg-k 51 \
  --log-dir ./apppredictor_training_logs \
  --log-name apppredictor_training \
  --checkpoint-dir ./apppredictor-training-checkpoints \
  --vit-model-name google/vit-base-patch16-224-in21k \
  --model-numbers 51 \
  --model-dim 768 \
  --hidden-dim 512 \
  --model-save-path ./apppredictor_model.ckpt \
  --vit-save-path ./vit_model.ckpt
```

### Evaluate proposed method

```bash
uv run python main.py evaluate-proposed-method \
  --seed 0 \
  --input-size 224 \
  --test-csv ./data/test/test.csv \
  --test-data-dir ./data/test \
  --model-path ./apppredictor_model.ckpt \
  --vit-path ./vit_model.ckpt \
  --results-excel-path ./apppredictor_evaluation_results.xlsx
```

### Perform Cross Validation with proposed method

```bash
uv run python main.py cross-validation \
  --seed 0 \
  --train-csv ./data/train/train.csv \
  --train-data-dir ./data/train \
  --alphas 0 --alphas 0.1 --alphas 0.2 --alphas 0.3 --alphas 0.4 --alphas 0.5 --alphas 0.6 --alphas 0.7 --alphas 0.8 --alphas 0.9 --alphas 1.0 \
  --neural-ndcg-ks 5 --neural-ndcg-ks 20 --neural-ndcg-ks 51 \
  --vit-model-name google/vit-base-patch16-224-in21k \
  --model-numbers 51 \
  --model-dim 768 \
  --hidden-dim 512 \
  --max-epochs 100 \
  --log-dir ./apppredictor_cv_logs \
  --log-name apppredictor_cv \
  --checkpoint-dir ./apppredictor-cv-checkpoints \
  --val-num 1 \
  --early-stopping-patience 10 \
  --lr 0.001 \
  --evaluates-results-excel ./apppredictor_cv_evaluation_results.xlsx \
  --test-results-excel ./apppredictor_cv_test_results.xlsx \
  --best-hyperparameters-excel ./apppredictor_cv_best_hyperparameters.xlsx
```
