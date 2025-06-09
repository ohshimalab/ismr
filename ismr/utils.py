import os
import random
from datetime import datetime

import lightly.transforms.utils as transforms_utils
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from lightly.transforms import SimSiamTransform
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel

from .baselines import FixedRankBaselineRetriever, SimSiamModel, SimSiamRetriever
from .dataset import CustomDataset
from .evaluator import Evaluator
from .models import APPredictor


def get_dcg(relevances: list[int], k: int = None, boost: bool = False) -> float:
    """
    Calculate the Discounted Cumulative Gain (DCG) at K.

    Args:
        relevances (list[int]): List of relevance scores, in ranked order.
        k (int, optional): Rank position to calculate DCG. If None, use the entire list.
        boost (bool, optional): If True, use exponential gain (2^rel - 1). If False, use linear gain.

    Returns:
        float: The DCG score.
    """
    relevances = np.array(relevances)
    if k is not None:
        relevances = relevances[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    if boost:
        gains = np.power(2, relevances) - 1
    else:
        gains = relevances
    return np.sum(gains / discounts)


def get_ndcg(relevances: list[int], k: int = None, boost: bool = False) -> float:
    """
    Calculate the Normalized Discounted Cumulative Gain (nDCG) at K.

    Args:
        relevances (list[int]): List of relevance scores, in the order that they are ranked.
        k (int, optional): Rank position to calculate nDCG. If None, use the entire list.
        boost (bool, optional): If True, use exponential gain (2^rel - 1). If False, use linear gain.

    Returns:
        float: The nDCG score.
    """
    if not relevances:
        return 0.0
    ideal_relevances = sorted(relevances, reverse=True)
    dcg_max = get_dcg(ideal_relevances, k, boost)
    if dcg_max == 0.0:
        return 0.0
    return get_dcg(relevances, k, boost) / dcg_max


def get_vit_fe_model(model_name: str = "google/vit-base-patch16-224-in21k") -> tuple:
    """Get the ViT feature extractor and model.

    Args:
        model_name (str, optional): Name of the ViT model to load. Defaults to "google/vit-base-patch16-224-in21k".

    Returns:
        tuple: A tuple containing the ViT feature extractor and model.
    """
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        model_name,
        do_resize=False,
        do_normalize=False,
    )
    model = ViTModel.from_pretrained(model_name)
    return feature_extractor, model


def seed_all(seed: int = 0):
    """Seed all random number generators for reproducibility.

    Args:
        seed (int, optional): Seed number. Defaults to 0.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(input_size: int = 224) -> tuple:
    """Get the data transforms for training and validation.

    Args:
        input_size (int, optional): Size of the input images. Defaults to 224.
    Returns:
        tuple: A tuple containing the training and validation transforms.
    """

    input_size = 224
    normalize = {
        "mean": transforms_utils.IMAGENET_NORMALIZE["mean"],
        "std": transforms_utils.IMAGENET_NORMALIZE["std"],
    }
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=normalize["mean"],
                std=normalize["std"],
            ),
        ]
    )
    val_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=normalize["mean"],
                std=normalize["std"],
            ),
        ]
    )
    return train_transforms, val_transforms


def get_current_dt_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_train_val_category_indicies(category_num: int, test_category_index: int, val_num: int):
    """
    Splits category indices into training and validation sets.

    The validation indices are chosen as the val_num contiguous indices immediately
    to the left of test_category_index. If there are not enough indices on the left in
    linear order, the selection will wrap around from the end.

    Parameters:
        category_num (int): Total number of indices (assumed to be 0 to category_num - 1).
        test_category_index (int): The index that is designated for testing.
        val_num (int): The number of contiguous indices to use for validation.
                       These indices are taken immediately before test_category_index;
                       if not enough exist, wrapping from the end fills in.

    Returns:
        tuple:
            train_category_indicies (list[int]): The indices not used in the validation set and not equal to test_category_index.
            val_category_indicies (list[int]): A list of exactly val_num indices selected as validation indices.

    Raises:
        ValueError: If category_num is not positive or test_category_index is out of valid range.
    """
    if category_num <= 0:
        raise ValueError("category_num must be positive.")
    if test_category_index < 0 or test_category_index >= category_num:
        raise ValueError("test_category_index must be between 0 and category_num - 1.")

    # Select val_num indices immediately to the left of test_category_index.
    # Use modular arithmetic to wrap around if necessary.
    val_category_indicies = [(test_category_index - val_num + i) % category_num for i in range(val_num)]

    # The training indices are all indices excluding the validation block and the test index.
    train_category_indicies = [
        i for i in range(category_num) if i not in val_category_indicies and i != test_category_index
    ]

    return train_category_indicies, val_category_indicies


def get_dataset(df: pd.DataFrame, root_dir: str, transform=None) -> CustomDataset:
    """Get the dataset.

    Args:
        df (pd.DataFrame): dataframe contains the data.
        root_dir (str): root directory of the images.
        transform: data transforms.

    Returns:
        CustomDataset: the dataset.
    """
    df = df.copy(deep=True)
    image_paths = []
    image_names = []
    image_categories = []
    model_scores = []
    model_relative_aps = []
    model_relevances = []
    for image_path in tqdm(df["Image Path"].unique().tolist()):
        image_paths.append(image_path)
        df_filtered = df[df["Image Path"] == image_path].sort_values("Model ID")
        df_filtered = df_filtered.sort_values(["Model ID"])
        image_name = df_filtered["File Name"].values[0]
        image_category = df_filtered["Category"].values[0]
        image_names.append(image_name)
        image_categories.append(image_category)
        model_score_list = df_filtered["AP"].values.tolist()
        model_scores.append(model_score_list)
        model_relative_aps.append(df_filtered["Relative AP"].values.tolist())
        model_relevances.append(df_filtered["Relevance Score"].values.tolist())
    dataset = CustomDataset(
        root_dir=root_dir,
        image_names=image_names,
        image_paths=image_paths,
        image_categories=image_categories,
        model_scores_lists=model_scores,
        model_relative_ap_lists=model_relative_aps,
        model_relevance_lists=model_relevances,
        transform=transform,
    )
    return dataset


def train_model(train_dataloader, config: dict, val_dataloader=None):
    """
    Train the APPredictor model.

    Args:
        train_dataloader: DataLoader for training data.
        config: Configuration dictionary containing model parameters.
          contains:
            - vit_model_name: Name of the ViT model to use.
            - model_numbers: Number of models to predict.
            - model_dim: Dimension of the model features.
            - hidden_dim: Dimension of the hidden layer.
            - lr: Learning rate for the optimizer.
            - alpha: Weight for the loss function.
            - neuralNDCG_k: k value for Neural NDCG calculation.
            - max_epochs: Number of epochs to train.
            - log_dir: Directory to save logs.
            - log_name: Name of the log file.
            - checkpoint_dir: Directory to save model checkpoints.
            - early_stopping_patience: Patience for early stopping (optional).
        val_dataloader: DataLoader for validation data (optional).


    Returns:
        trainer: Trained PyTorch Lightning Trainer instance.
        ap_predictor: Trained APPredictor model instance.
    """
    _, vit_model_instance = get_vit_fe_model(
        model_name=config["vit_model_name"],
    )
    ap_predictor = APPredictor(
        vit=vit_model_instance,
        model_numbers=config["model_numbers"],
        model_dim=config["model_dim"],
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        alpha=config["alpha"],
        neuralNDCG_k=config["neuralNDCG_k"],
    )
    callbacks = []
    if "early_stopping_patience" in config and config["early_stopping_patience"] > 0 and val_dataloader is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=config["early_stopping_patience"],
                mode="min",
            )
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=config["checkpoint_dir"],
                filename="appredictor-{epoch:02d}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )
        )
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices=1,
        logger=CSVLogger(
            save_dir=config["log_dir"],
            name=config["log_name"],
        ),
        callbacks=callbacks,
    )
    if val_dataloader is not None:
        trainer.fit(
            model=ap_predictor,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    else:
        trainer.fit(
            model=ap_predictor,
            train_dataloaders=train_dataloader,
        )
    return trainer, ap_predictor


def get_train_val_category_indicies(category_num: int, test_category_index: int, val_num: int):
    """
    Splits category indices into training and validation sets.

    The validation indices are chosen as the val_num contiguous indices immediately
    to the left of test_category_index. If there are not enough indices on the left in
    linear order, the selection will wrap around from the end.

    Parameters:
        category_num (int): Total number of indices (assumed to be 0 to category_num - 1).
        test_category_index (int): The index that is designated for testing.
        val_num (int): The number of contiguous indices to use for validation.
                       These indices are taken immediately before test_category_index;
                       if not enough exist, wrapping from the end fills in.

    Returns:
        tuple:
            train_category_indicies (list[int]): The indices not used in the validation set and not equal to test_category_index.
            val_category_indicies (list[int]): A list of exactly val_num indices selected as validation indices.

    Raises:
        ValueError: If category_num is not positive or test_category_index is out of valid range.
    """
    if category_num <= 0:
        raise ValueError("category_num must be positive.")
    if test_category_index < 0 or test_category_index >= category_num:
        raise ValueError("test_category_index must be between 0 and category_num - 1.")

    # Select val_num indices immediately to the left of test_category_index.
    val_category_indicies = [(test_category_index - val_num + i) % category_num for i in range(val_num)]

    # The training indices are all indices excluding the validation block and the test index.
    train_category_indicies = [
        i for i in range(category_num) if i not in val_category_indicies and i != test_category_index
    ]

    return train_category_indicies, val_category_indicies


def evaluate(
    model,
    dataset,
    test_df,
):
    """
    Evaluate the model on the test dataset

    Args:
        model: Trained APPredictor model.
        dataset: Test dataset.
        test_df: DataFrame containing test metadata.
        output_csv_path: Path to save the evaluation results CSV.
    """
    # Extract test data
    test_items = [item for item in dataset]
    categories = [item[0] for item in test_items]
    image_names = [item[1] for item in test_items]
    image_tensors = [item[2] for item in test_items]
    ap_relative_lists = [item[4] for item in test_items]
    ap_score_lists = [item[3] for item in test_items]
    image_tensors = torch.stack(image_tensors).cpu()

    # Prepare ground truth relevance matrix
    model_ids = sorted(test_df["Model ID"].unique().tolist())
    file_names = sorted(test_df["File Name"].unique().tolist())
    pivot_rel = (
        test_df.pivot_table(
            index="File Name",
            columns="Model ID",
            values="Relevance Score",
        )
        .fillna(0)
        .reindex(columns=model_ids, fill_value=0)
        .reindex(index=file_names)
    )
    gt_rel_matrix = pivot_rel.values
    image_id_list = pivot_rel.index.tolist()

    evaluator = Evaluator(gt_rel_matrix, image_id_list)

    # Predict scores
    with torch.no_grad():
        predicted_scores = model(image_tensors).cpu().numpy()

    # Evaluate and collect results
    results = []
    for idx, img_id in enumerate(image_names):
        img_scores = predicted_scores[idx, :].tolist()
        ap_relative = ap_relative_lists[idx]
        ap_score = ap_score_lists[idx]
        metrics = evaluator.evaluate(
            img_id,
            img_scores,
            ap_relative=ap_relative,
            ap_score=ap_score,
            pred_ap_score=img_scores,
        )
        row = {
            "Image ID": img_id,
            "Category": categories[idx],
        }
        row.update(metrics)
        results.append(row)

    results_df = pd.DataFrame(results)
    return results_df


def cross_validate(
    df: pd.DataFrame,
    alphas: list[float],
    neuralNDCG_ks: list[int],
    config: dict,
):
    """
    Perform cross-validation over categories, searching for best hyperparameters.

    Args:
        df (pd.DataFrame): Input dataframe.
        alphas (list[float]): List of alpha values to try.
        neuralNDCG_ks (list[int]): List of neuralNDCG_k values to try.
        config (dict): Configuration dictionary.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (all_cv_results, best_hyperparams, test_results)
    """
    train_transform, val_transform = get_transforms(input_size=224)
    all_cv_results = []
    best_hyperparams = []
    all_test_results = []
    categories = df["Category"].unique().tolist()

    def get_category_splits(categories, test_idx, val_num):
        val_indices = [(test_idx - val_num + i) % len(categories) for i in range(val_num)]
        train_indices = [i for i in range(len(categories)) if i not in val_indices and i != test_idx]
        return [categories[i] for i in train_indices], [categories[i] for i in val_indices]

    for test_idx, test_category in enumerate(categories):
        train_cats, val_cats = get_category_splits(categories, test_idx, config["val_num"])
        train_df = df[df["Category"].isin(train_cats)].sort_values(["File Name", "Model ID"]).reset_index(drop=True)
        val_df = df[df["Category"].isin(val_cats)].reset_index(drop=True)
        test_df = df[(df["Category"] == test_category) & (df["Image Type"] == "original")].reset_index(drop=True)

        train_dataset = get_dataset(train_df, root_dir=config["data_root_dir"], transform=train_transform)
        val_dataset = get_dataset(val_df, root_dir=config["data_root_dir"], transform=val_transform)
        test_dataset = get_dataset(test_df, root_dir=config["data_root_dir"], transform=val_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

        # Hyperparameter search
        for alpha in alphas:
            for ndcg_k in neuralNDCG_ks:
                run_cfg = config.copy()
                run_cfg.update(
                    {
                        "alpha": alpha,
                        "neuralNDCG_k": ndcg_k,
                        "log_dir": f"./logs/{test_category}_{alpha}_{ndcg_k}",
                        "log_name": f"{test_category}_{alpha}_{ndcg_k}.csv",
                        "checkpoint_dir": f"./checkpoints/{test_category}_{alpha}_{ndcg_k}",
                    }
                )
                trainer, model = train_model(
                    train_dataloader=train_loader,
                    config=run_cfg,
                    val_dataloader=val_loader,
                )
                for epoch in range(len(model.train_epoch_losses["mse"])):
                    all_cv_results.append(
                        {
                            "epoch": epoch,
                            "test_category": test_category,
                            "train_val": "train",
                            "alpha": alpha,
                            "neuralNDCG_k": ndcg_k,
                            "mse": model.train_epoch_losses["mse"][epoch],
                            "l2r": model.train_epoch_losses["l2r"][epoch],
                            "ndcg_3": model.train_epoch_ndcg[3][epoch],
                            "ndcg_5": model.train_epoch_ndcg[5][epoch],
                        }
                    )
                    all_cv_results.append(
                        {
                            "epoch": epoch,
                            "test_category": test_category,
                            "train_val": "val",
                            "alpha": alpha,
                            "neuralNDCG_k": ndcg_k,
                            "mse": model.val_epoch_losses["mse"][epoch],
                            "l2r": model.val_epoch_losses["l2r"][epoch],
                            "ndcg_3": model.val_epoch_ndcg[3][epoch],
                            "ndcg_5": model.val_epoch_ndcg[5][epoch],
                        }
                    )
                del model
                del trainer
                torch.cuda.empty_cache()

        # Select best hyperparameters based on validation ndcg_avg
        cv_df = pd.DataFrame(
            [r for r in all_cv_results if r["test_category"] == test_category and r["train_val"] == "val"]
        )
        cv_df["ndcg_avg"] = (cv_df["ndcg_3"] + cv_df["ndcg_5"]) / 2
        best_row = cv_df.loc[cv_df["ndcg_avg"].idxmax()]
        best_alpha = float(best_row["alpha"])
        best_ndcg_k = int(best_row["neuralNDCG_k"])
        best_epoch = int(best_row["epoch"]) + 1

        print(f"Best for {test_category}: alpha={best_alpha}, ndcg_k={best_ndcg_k}, epoch={best_epoch}")

        # Retrain with best hyperparameters and evaluate on test set
        best_cfg = config.copy()
        best_cfg.update(
            {
                "alpha": best_alpha,
                "neuralNDCG_k": best_ndcg_k,
                "log_dir": f"./logs/{test_category}_{best_alpha}_{best_ndcg_k}",
                "log_name": f"{test_category}_{best_alpha}_{best_ndcg_k}.csv",
                "checkpoint_dir": f"./checkpoints/{test_category}_{best_alpha}_{best_ndcg_k}",
                "max_epochs": best_epoch,
            }
        )
        trainer, model = train_model(
            train_dataloader=train_loader,
            config=best_cfg,
        )
        test_eval = evaluate(
            model=model,
            dataset=test_dataset,
            test_df=test_df,
        )
        best_hyperparams.append(
            {
                "test_category": test_category,
                "best_alpha": best_alpha,
                "best_neuralNDCG_k": best_ndcg_k,
                "best_epoch": best_epoch,
            }
        )
        all_test_results.append(test_eval)

    all_cv_results_df = pd.DataFrame(all_cv_results)
    return (
        all_cv_results_df,
        pd.DataFrame(best_hyperparams),
        pd.concat(all_test_results, ignore_index=True),
    )


def get_embeddings_by_simsiam_model(dataloader: DataLoader, model: SimSiamModel, device: str):
    """Generates embeddings for all images using the model's backbone.

    Args:
        dataloader (DataLoader): DataLoader with basic transforms (no SimSiam augmentations).
        model (SimSiamModel): The trained SimSiam model.
        device (str): Device for inference ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Tensor of all embeddings. Shape: (num_total_images, num_ftrs)
    """
    embeddings_list = []
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            # batch[2] is the image tensor x. Shape: (batch_size, C, H, W)
            x = batch[2].to(device)

            # Backbone features (before SimSiam projection head)
            # y shape: (batch_size, num_ftrs)
            y = model.backbone(pixel_values=x).last_hidden_state[:, 0, :]
            embeddings_list.append(y.cpu())

    all_embeddings = torch.cat(embeddings_list, dim=0)
    # all_embeddings shape: (total_num_images, num_ftrs)
    return all_embeddings


def get_simsiam_transforms(input_size: int = 224) -> tuple:
    """Get the data transforms for SimSiam training.

    Args:
        input_size (int, optional): Size of the input images. Defaults to 224.

    Returns:
        tuple: A tuple containing the training and validation transforms.
    """
    normalize = {
        "mean": transforms_utils.IMAGENET_NORMALIZE["mean"],
        "std": transforms_utils.IMAGENET_NORMALIZE["std"],
    }
    train_transforms = SimSiamTransform(input_size=input_size, normalize=normalize)
    val_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=normalize["mean"],
                std=normalize["std"],
            ),
        ]
    )
    return train_transforms, val_transforms


def train_simsiam_model(
    dataloader: DataLoader,
    config: dict,
    val_dataloader: DataLoader = None,
    device: str = "cuda",
) -> tuple[pl.Trainer, SimSiamModel]:
    """Train the SimSiam model.

    Args:
        dataloader (DataLoader): DataLoader for training data.
        config (dict): Configuration dictionary containing model parameters.
        val_dataloader (DataLoader, optional): DataLoader for validation data. Defaults to None.
        device (str, optional): Device to use for training ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        tuple: A tuple containing the trained PyTorch Lightning Trainer instance and the SimSiam model.
    """
    _, vit_backbone = get_vit_fe_model(
        model_name=config["vit_model_name"],
    )
    simsiam_model = SimSiamModel(
        backbone=vit_backbone,
        num_ftrs=config["num_ftrs"],
        proj_hidden_dim=config["proj_hidden_dim"],
        pred_hidden_dim=config["pred_hidden_dim"],
        out_dim=config["out_dim"],
        learning_rate=config["lr"],
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=config["checkpoint_dir"],
            filename="simsiam-{epoch:02d}",
        ),
    ]

    if "early_stopping_patience" in config and config["early_stopping_patience"] > 0 and val_dataloader is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=config["early_stopping_patience"],
                mode="min",
            )
        )

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices=1 if device == "cuda" else None,
        logger=CSVLogger(
            save_dir=config["log_dir"],
            name=config["log_name"],
        ),
        callbacks=callbacks,
    )

    if val_dataloader is not None:
        trainer.fit(
            model=simsiam_model,
            train_dataloaders=dataloader,
            val_dataloaders=val_dataloader,
        )
    else:
        trainer.fit(
            model=simsiam_model,
            train_dataloaders=dataloader,
        )

    return trainer, simsiam_model


def evaluate_simsiam_model(
    model: SimSiamModel,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Evaluate the SimSiam model using retrieval and ranking metrics.

    Args:
        model (SimSiamModel): Trained SimSiam model.
        train_dataloader (DataLoader): DataLoader for training images (to build pivots).
        test_dataloader (DataLoader): DataLoader for test images.
        train_df (pd.DataFrame): DataFrame with training metadata.
        test_df (pd.DataFrame): DataFrame with test metadata.
        config (dict): Configuration dictionary.

    Returns:
        pd.DataFrame: DataFrame with evaluation results for each test image.
    """
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    train_df = train_df.copy(deep=True).sort_values(["Category", "File Name", "Model ID"]).reset_index(drop=True)
    test_df = test_df.copy(deep=True).sort_values(["Category", "File Name", "Model ID"]).reset_index(drop=True)

    def get_normalized_embeddings(dataloader, model, device):
        """Helper to get normalized embeddings from a dataloader."""
        embeddings = get_embeddings_by_simsiam_model(dataloader, model, device)
        return F.normalize(embeddings, dim=1)

    # Compute normalized embeddings for pivots (train) and test
    pivot_emb = get_normalized_embeddings(train_dataloader, model, device)
    test_emb = get_normalized_embeddings(test_dataloader, model, device)

    # Prepare AP matrix and IDs for retrieval
    ap_matrix_df = train_df.pivot_table(values="AP", index="Model ID", columns=["Category", "File Name"])
    ap_matrix = ap_matrix_df.values
    model_ids = ap_matrix_df.index.tolist()

    # Build retriever
    simsiam_retriever = SimSiamRetriever(
        pivot_embeddings=pivot_emb.numpy(),
        image_model_ap_matrix=ap_matrix,
        model_ids=model_ids,
        faiss_top_k_search=config.get("faiss_top_k_search", None),
    )

    # Prepare ground truth relevance matrix for evaluation
    test_file_names = sorted(test_df["File Name"].unique().tolist())
    relevance_pivot = (
        test_df.pivot_table(
            index="File Name",
            columns="Model ID",
            values="Relevance Score",
        )
        .fillna(0)
        .reindex(columns=model_ids, fill_value=0)
        .reindex(index=test_file_names)
    )
    gt_rel_matrix = relevance_pivot.values

    evaluator = Evaluator(gt_rel_matrix, test_file_names)

    # Retrieve model scores for test images
    _, retrieved_scores = simsiam_retriever.retrieve(test_emb)

    # Evaluate each test image
    results = []
    test_dataset = test_dataloader.dataset
    for idx, img_id in enumerate(test_dataset.image_names):
        pred_scores = retrieved_scores[idx, :].cpu().numpy().tolist()
        ap_relative = test_dataset.model_relative_ap_lists[idx]
        metrics = evaluator.evaluate(
            img_id,
            pred_scores,
            pred_ap_score=pred_scores,
            ap_relative=ap_relative,
        )
        row = {
            "Image ID": img_id,
            "Category": test_dataset.image_categories[idx],
        }
        row.update(metrics)
        results.append(row)

    return pd.DataFrame(results)


def evaluate_fixed_rank(
    retriever: FixedRankBaselineRetriever,
    dataset: CustomDataset,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate a fixed-rank retriever on a test dataset.

    Args:
        retriever (FixedRankBaselineRetriever): The fixed-rank retriever.
        dataset (CustomDataset): The test dataset.
        meta_df (pd.DataFrame): DataFrame containing test metadata.

    Returns:
        pd.DataFrame: DataFrame with evaluation results for each test image.
    """
    # Extract test data
    items = [item for item in dataset]
    categories = [item[0] for item in items]
    image_names = [item[1] for item in items]
    ap_relative_lists = [item[4] for item in items]

    # Prepare ground truth relevance matrix
    model_ids = sorted(meta_df["Model ID"].unique())
    file_names = sorted(meta_df["File Name"].unique())
    relevance_matrix = (
        meta_df.pivot_table(
            index="File Name",
            columns="Model ID",
            values="Relevance Score",
        )
        .fillna(0)
        .reindex(columns=model_ids, fill_value=0)
        .reindex(index=file_names)
        .values
    )
    evaluator = Evaluator(relevance_matrix, file_names)

    # Evaluate and collect results
    results = []
    for idx, img_id in enumerate(image_names):
        scores = retriever.fixed_scores.tolist()
        ap_relative = ap_relative_lists[idx]
        metrics = evaluator.evaluate(
            img_id,
            scores,
            ap_relative=ap_relative,
        )
        row = {
            "Image ID": img_id,
            "Category": categories[idx],
        }
        row.update(metrics)
        results.append(row)

    return pd.DataFrame(results)
