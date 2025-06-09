import pickle

import pandas as pd
import pytorch_lightning as pl
import torch
import typer
from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rich import print
from torch.utils.data import DataLoader
from typing_extensions import Annotated

from ismr import utils
from ismr.baselines import FixedRankBaselineRetriever, SimSiamModel
from ismr.evaluator import Evaluator
from ismr.models import APPredictor

app = typer.Typer()


@app.command()
def train_proposed_method(
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 0,
    input_size: Annotated[int, typer.Option("--input-size", help="Image Input Size")] = 224,
    train_csv: Annotated[
        str, typer.Option("--train-csv", help="Path to the training CSV file")
    ] = "./data/train/train.csv",
    train_data_dir: Annotated[
        str, typer.Option("--train-data-dir", help="Directory containing training images")
    ] = "./data/train",
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size for training")] = 32,
    num_workers: Annotated[int, typer.Option("--num-workers", help="Number of workers for DataLoader")] = 4,
    max_epochs: Annotated[int, typer.Option("--max-epochs", help="Number of max epochs to train")] = 100,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate for the optimizer")] = 0.001,
    alpha: Annotated[float, typer.Option("--alpha", help="Alpha parameter for the model")] = 0.5,
    neuralNDCG_k: Annotated[int, typer.Option("--neural-ndcg-k", help="k for Neural NDCG")] = 51,
    log_dir: Annotated[str, typer.Option("--log-dir", help="Directory to save logs")] = "./apppredictor_training_logs",
    log_name: Annotated[str, typer.Option("--log-name", help="Name for the log file")] = "apppredictor_training",
    checkpoint_dir: Annotated[
        str, typer.Option("--checkpoint-dir", help="Directory to save checkpoints")
    ] = "./apppredictor-training-checkpoints",
    vit_model_name: Annotated[
        str, typer.Option("--vit-model-name", help="Name of the ViT model")
    ] = "google/vit-base-patch16-224-in21k",
    model_numbers: Annotated[int, typer.Option("--model-numbers", help="Number of target models")] = 51,
    model_dim: Annotated[int, typer.Option("--model-dim", help="Dimension of the model vectors")] = 768,
    hidden_dim: Annotated[int, typer.Option("--hidden-dim", help="Hidden dimension for the MLP")] = 512,
    model_save_path: Annotated[
        str, typer.Option("--model-save-path", help="Path to save the trained model")
    ] = "./apppredictor_model.ckpt",
    vit_save_path: Annotated[
        str, typer.Option("--vit-save-path", help="Path to save the ViT model")
    ] = "./vit_model.ckpt",
):
    utils.seed_all(seed=seed)
    train_transform, val_transform = utils.get_transforms(input_size=input_size)
    train_df = pd.read_csv(train_csv)
    train_dataset = utils.get_dataset(
        train_df,
        train_data_dir,
        transform=train_transform,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    trainer, ap_predictor = utils.train_model(
        train_dataloader,
        config={
            "vit_model_name": vit_model_name,
            "model_numbers": model_numbers,
            "model_dim": model_dim,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "alpha": alpha,
            "neuralNDCG_k": neuralNDCG_k,
            "max_epochs": max_epochs,
            "log_dir": log_dir,
            "log_name": log_name,
            "checkpoint_dir": checkpoint_dir,
        },
    )
    trainer.save_checkpoint(model_save_path)
    with open(vit_save_path, "wb") as f:
        pickle.dump(ap_predictor.vit, f)
    print(f"ViT model saved to {vit_save_path}")
    print(f"Model saved to {model_save_path}")


@app.command()
def evaluate_proposed_method(
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 0,
    input_size: Annotated[int, typer.Option("--input-size", help="Input size for the model")] = 224,
    test_csv: Annotated[str, typer.Option("--test-csv", help="Path to the testing CSV file")] = "./data/test/test.csv",
    test_data_dir: Annotated[
        str, typer.Option("--test-data-dir", help="Directory containing test images")
    ] = "./data/test",
    model_path: Annotated[
        str, typer.Option("--model-path", help="Path to the trained model checkpoint")
    ] = "./apppredictor_model.ckpt",
    vit_path: Annotated[str, typer.Option("--vit-path", help="Path to the ViT model checkpoint")] = "./vit_model.ckpt",
    results_excel_path: Annotated[
        str, typer.Option("--results-excel-path", help="Path to save the evaluation results Excel file")
    ] = "./apppredictor_evaluation_results.xlsx",
):
    utils.seed_all(seed)
    _, val_transform = utils.get_transforms(input_size=input_size)
    test_df = pd.read_csv(test_csv)
    test_dataset = utils.get_dataset(
        test_df,
        test_data_dir,
        transform=val_transform,
    )
    with open(vit_path, "rb") as f:
        vit_model = pickle.load(f)

    ap_predictor = APPredictor.load_from_checkpoint(
        model_path,
        vit=vit_model,
    )
    ap_predictor.eval()
    results = utils.evaluate(model=ap_predictor, dataset=test_dataset, test_df=test_df)
    results.to_excel(results_excel_path, index=False)


@app.command()
def cross_validation(
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility")] = 0,
    train_csv: Annotated[
        str, typer.Option("--train-csv", help="Path to the training CSV file")
    ] = "./data/train/train.csv",
    train_data_dir: Annotated[
        str, typer.Option("--train-data-dir", help="Directory containing training images")
    ] = "./data/train",
    alphas: Annotated[list[float], typer.Option("--alphas", help="List of alpha values for cross-validation")] = [
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    neuralNDCG_ks: Annotated[list[int], typer.Option("--neural-ndcg-ks", help="List of k values for Neural NDCG")] = [
        5,
        20,
        51,
    ],
    vit_model_name: Annotated[
        str, typer.Option("--vit-model-name", help="Name of the ViT model")
    ] = "google/vit-base-patch16-224-in21k",
    model_numbers: Annotated[int, typer.Option("--model-numbers", help="Number of target models")] = 51,
    model_dim: Annotated[int, typer.Option("--model-dim", help="Dimension of the model vectors")] = 768,
    hidden_dim: Annotated[int, typer.Option("--hidden-dim", help="Hidden dimension for the MLP")] = 512,
    max_epochs: Annotated[int, typer.Option("--max-epochs", help="Number of max epochs to train")] = 100,
    log_dir: Annotated[str, typer.Option("--log-dir", help="Directory to save logs")] = "./apppredictor_cv_logs",
    log_name: Annotated[str, typer.Option("--log-name", help="Name for the log file")] = "apppredictor_cv",
    checkpoint_dir: Annotated[
        str, typer.Option("--checkpoint-dir", help="Directory to save checkpoints")
    ] = "./apppredictor-cv-checkpoints",
    val_num: Annotated[
        int, typer.Option("--val-num", help="Number of validation categories for each fold in cross-validation")
    ] = 1,
    early_stopping_patience: Annotated[
        int, typer.Option("--early-stopping-patience", help="Early stopping patience")
    ] = 10,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate for the optimizer")] = 0.001,
    evaluates_results_excel: Annotated[
        str, typer.Option("--evaluates-results-excel", help="Path to save the evaluation results Excel file")
    ] = "./apppredictor_cv_evaluation_results.xlsx",
    test_results_excel: Annotated[
        str, typer.Option("--test-results-excel", help="Path to save the test results Excel file")
    ] = "./apppredictor_cv_test_results.xlsx",
    best_hyperparameters_excel: Annotated[
        str, typer.Option("--best-hyperparameters-excel", help="Path to save the best hyperparameters Excel file")
    ] = "./apppredictor_cv_best_hyperparameters.xlsx",
):
    utils.seed_all(seed=seed)
    train_df = pd.read_csv(train_csv)
    cross_validation_evalutes_df, best_hyperparameters, test_results = utils.cross_validate(
        df=train_df,
        alphas=alphas,
        neuralNDCG_ks=neuralNDCG_ks,
        config={
            "vit_model_name": vit_model_name,
            "model_numbers": model_numbers,
            "model_dim": model_dim,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "max_epochs": max_epochs,
            "log_dir": log_dir,
            "log_name": log_name,
            "checkpoint_dir": checkpoint_dir,
            "val_num": val_num,
            "early_stopping_patience": early_stopping_patience,
            "data_root_dir": train_data_dir,
        },
    )
    cross_validation_evalutes_df.to_excel(evaluates_results_excel, index=False)
    best_hyperparameters.to_excel(best_hyperparameters_excel, index=False)
    test_results.to_excel(test_results_excel, index=False)


@app.command()
def train_simsiam(
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility")] = 0,
    input_size: Annotated[int, typer.Option("--input-size", help="Input size for the model")] = 224,
    max_epochs: Annotated[int, typer.Option("--max-epochs", help="Number of max epochs to train")] = 30,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size for training")] = 32,
    num_workers: Annotated[int, typer.Option("--num-workers", help="Number of workers for DataLoader")] = 4,
    log_dir: Annotated[str, typer.Option("--log-dir", help="Directory to save logs")] = "./simsiam_training_logs",
    log_name: Annotated[str, typer.Option("--log-name", help="Name for the log file")] = "simsiam_training",
    checkpoint_dir: Annotated[
        str, typer.Option("--checkpoint-dir", help="Directory to save checkpoints")
    ] = "./simsiam-training-checkpoints",
    vit_model_name: Annotated[
        str, typer.Option("--vit-model-name", help="Name of the ViT model")
    ] = "google/vit-base-patch16-224-in21k",
    num_ftrs: Annotated[int, typer.Option("--num-ftrs", help="Number of features for the model")] = 768,
    proj_hidden_dim: Annotated[int, typer.Option("--proj-hidden-dim", help="Projection hidden dimension")] = 2048,
    pred_hidden_dim: Annotated[int, typer.Option("--pred-hidden-dim", help="Prediction hidden dimension")] = 512,
    out_dim: Annotated[int, typer.Option("--out-dim", help="Output dimension for the model")] = 768,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate for the optimizer")] = 0.001,
    train_csv: Annotated[
        str, typer.Option("--train-csv", help="Path to the training CSV file")
    ] = "./data/train/train.csv",
    train_data_dir: Annotated[
        str, typer.Option("--train-data-dir", help="Directory containing training images")
    ] = "./data/train",
    model_save_path: Annotated[
        str, typer.Option("--model-save-path", help="Path to save the trained SimSiam model")
    ] = "./simsiam_model.ckpt",
    vit_save_path: Annotated[
        str, typer.Option("--vit-save-path", help="Path to save the ViT model")
    ] = "./vit_model.ckpt",
):
    utils.seed_all(seed)
    train_transform, _ = utils.get_simsiam_transforms(input_size=input_size)
    train_df = pd.read_csv(train_csv)
    train_dataset = utils.get_dataset(
        train_df,
        train_data_dir,
        transform=train_transform,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    trainer, simsiam_model = utils.train_simsiam_model(
        train_dataloader,
        config={
            "vit_model_name": vit_model_name,
            "num_ftrs": num_ftrs,
            "proj_hidden_dim": proj_hidden_dim,
            "pred_hidden_dim": pred_hidden_dim,
            "out_dim": out_dim,
            "lr": lr,
            "max_epochs": max_epochs,
            "log_dir": log_dir,
            "log_name": log_name,
            "checkpoint_dir": checkpoint_dir,
        },
    )
    trainer.save_checkpoint(model_save_path)
    with open(vit_save_path, "wb") as f:
        pickle.dump(simsiam_model.backbone, f)
    print(f"ViT model saved to {vit_save_path}")
    print(f"SimSiam model saved to {model_save_path}")


@app.command()
def evaluate_simsiam(
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility")] = 0,
    input_size: Annotated[int, typer.Option("--input-size", help="Input size for the model")] = 224,
    simsiam_model: Annotated[
        str, typer.Option("--simsiam-model", help="Path to the trained SimSiam model")
    ] = "./simsiam_model.ckpt",
    vit_model: Annotated[str, typer.Option("--vit-model", help="Path to the ViT model")] = "./vit_model.ckpt",
    train_csv: Annotated[
        str, typer.Option("--train-csv", help="Path to the training CSV file")
    ] = "./data/train/train.csv",
    train_data_dir: Annotated[
        str, typer.Option("--train-data-dir", help="Directory containing training images")
    ] = "./data/train",
    test_csv: Annotated[str, typer.Option("--test-csv", help="Path to the testing CSV file")] = "./data/test/test.csv",
    test_data_dir: Annotated[
        str, typer.Option("--test-data-dir", help="Directory containing test images")
    ] = "./data/test",
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size for evaluation")] = 32,
    num_workers: Annotated[int, typer.Option("--num-workers", help="Number of workers for DataLoader")] = 4,
    results_excel_path: Annotated[
        str, typer.Option("--results-excel-path", help="Path to save the evaluation results Excel file")
    ] = "./simsiam_evaluation_results.xlsx",
):
    utils.seed_all(seed)
    _, val_transform = utils.get_simsiam_transforms(input_size=input_size)
    train_df = pd.read_csv(train_csv)
    train_dataset = utils.get_dataset(train_df, train_data_dir, transform=val_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_df = pd.read_csv(test_csv)
    test_dataset = utils.get_dataset(
        test_df,
        test_data_dir,
        transform=val_transform,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    with open(vit_model, "rb") as f:
        vit_model = pickle.load(f)

    simsiam_model = SimSiamModel.load_from_checkpoint(
        simsiam_model,
        backbone=vit_model,
    )

    results_df = utils.evaluate_simsiam_model(
        model=simsiam_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        train_df=train_df,
        test_df=train_df,
        config={"device": "cuda" if torch.cuda.is_available() else "cpu", "faiss_top_k_search": 1},
    )
    results_df.to_excel(results_excel_path, index=False)


@app.command()
def evaluate_fixed_rank(
    train_csv: Annotated[str, typer.Option("--train-csv", help="Path to the training CSV file")],
    test_csv: Annotated[str, typer.Option("--test-csv", help="Path to the test CSV file")],
    test_data_dir: Annotated[str, typer.Option("--test-data-dir", help="Directory containing test images")],
    output_excel: Annotated[str, typer.Option("--output-excel", help="Path to save the results Excel file")],
):
    utils.seed_all(0)
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    test_dataset = utils.get_dataset(
        test_df,
        test_data_dir,
    )
    fixed_rank_predictor = FixedRankBaselineRetriever(train_df=train_df)
    results = utils.evaluate_fixed_rank(
        retriever=fixed_rank_predictor,
        dataset=test_dataset,
        meta_df=test_df,
    )
    results.to_excel(output_excel, index=False)


def main():
    app()


if __name__ == "__main__":
    main()
