from abc import ABC, abstractmethod

import faiss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead
from torch import nn


class SimSiamModel(pl.LightningModule):
    """SimSiam model implemented using PyTorch Lightning."""

    def __init__(
        self,
        backbone: nn.Module,
        num_ftrs: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        out_dim: int,
        learning_rate: float,
    ):
        """
        Args:
            backbone (nn.Module): The backbone network (e.g., ViT).
            num_ftrs (int): Number of features from the backbone.
            proj_hidden_dim (int): Hidden dimension of the projection head.
            pred_hidden_dim (int): Hidden dimension of the prediction head.
            out_dim (int): Output dimension of the projection/prediction heads.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])  # Avoid saving backbone state in hparams
        self.lr = learning_rate
        self.out_dim = out_dim

        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(num_ftrs, proj_hidden_dim, out_dim)
        self.prediction_head = SimSiamPredictionHead(out_dim, pred_hidden_dim, out_dim)
        self.criterion = NegativeCosineSimilarity()

        self.train_epoch_losses: list[float] = []
        self._train_batch_losses: list[float] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the SimSiam model.

        Args:
            x (torch.Tensor): Input image tensor.
                Shape: (batch_size, channels, height, width)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - z (torch.Tensor): Projected features (detached).
                  Shape: (batch_size, out_dim)
                - p (torch.Tensor): Predicted features.
                  Shape: (batch_size, out_dim)
        """
        # ViT backbone processing: extract [CLS] token embedding
        f = self.backbone(pixel_values=x).last_hidden_state[:, 0, :]
        # f shape: (batch_size, num_ftrs)

        z = self.projection_head(f)  # z shape: (batch_size, out_dim)
        p = self.prediction_head(z)  # p shape: (batch_size, out_dim)
        z = z.detach()  # Stop gradient for z in loss calculation (D(p, z.detach()))
        return z, p

    def _calculate_loss(self, z0: torch.Tensor, p0: torch.Tensor, z1: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        """Calculates the SimSiam loss.

        Args:
            z0 (torch.Tensor): Projected features of view 0. Shape: (batch_size, out_dim)
            p0 (torch.Tensor): Predicted features of view 0. Shape: (batch_size, out_dim)
            z1 (torch.Tensor): Projected features of view 1. Shape: (batch_size, out_dim)
            p1 (torch.Tensor): Predicted features of view 1. Shape: (batch_size, out_dim)

        Returns:
            torch.Tensor: The calculated loss (scalar).
        """
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        # loss shape: () (scalar tensor)
        return loss

    def training_step(self, batch: tuple[any, ...], batch_idx: int) -> torch.Tensor:
        """Performs a single training step.

        Args:
            batch (Tuple[Any, ...]): Output from CustomImageDataset.
                Contains (..., (view0, view1), ...).
                view0, view1 shape: (batch_size, channels, height, width)
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss for this training step.
        """
        (x0, x1) = batch[2]  # x0, x1 are augmented views. Shape: (batch_size, C, H, W)

        z0, p0 = self.forward(x0)  # z0, p0 shape: (batch_size, out_dim)
        z1, p1 = self.forward(x1)  # z1, p1 shape: (batch_size, out_dim)

        loss = self._calculate_loss(z0, z1, p0, p1)
        self.log(
            "train_loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self._train_batch_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self) -> None:
        """Hook called at the end of the training epoch."""
        if self._train_batch_losses:
            avg_loss = sum(self._train_batch_losses) / len(self._train_batch_losses)
            self.train_epoch_losses.append(avg_loss)
            self.log(
                "train_loss_epoch",
                avg_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self._train_batch_losses = []  # Reset for next epoch

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        # SimSiam paper uses SGD with specific settings. AdamW is a robust default.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, batch_images_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class SimSiamRetriever(Retriever):
    def __init__(
        self,
        pivot_embeddings: np.ndarray,
        image_model_ap_matrix: np.ndarray,
        model_ids: list[any],
        faiss_top_k_search: int = None,
    ):
        if pivot_embeddings.shape[0] != image_model_ap_matrix.shape[1]:
            raise ValueError("Number of pivot embeddings must match columns in AP matrix (N_pivot).")
        if len(model_ids) != image_model_ap_matrix.shape[0]:
            raise ValueError("Length of model_ids must match rows in AP matrix (N_models).")

        self.pivot_embeddings_np = pivot_embeddings.astype(np.float32)
        self.ap_matrix_np = image_model_ap_matrix.astype(np.float32)

        self.original_model_ids = list(model_ids)
        self.model_ids_tensor = torch.arange(len(self.original_model_ids), dtype=torch.long)

        self.num_models = image_model_ap_matrix.shape[0]
        self.embedding_dim = pivot_embeddings.shape[1]
        self.faiss_top_k_search = faiss_top_k_search if faiss_top_k_search is not None else pivot_embeddings.shape[0]

        print("SimSiamRetriever: Building FAISS index with pivot embeddings...")
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(self.pivot_embeddings_np)
        print(f"SimSiamRetriever: FAISS index built. Total items in index: {self.faiss_index.ntotal}")

    def _get_similarity_to_pivots(self, batch_test_embeddings_np: np.ndarray) -> np.ndarray:
        num_queries = batch_test_embeddings_np.shape[0]
        num_pivot_images = self.pivot_embeddings_np.shape[0]
        D_faiss, I_faiss = self.faiss_index.search(
            batch_test_embeddings_np.astype(np.float32), k=self.faiss_top_k_search
        )
        S_batch_pivot = np.zeros((num_queries, num_pivot_images), dtype=np.float32)
        for query_idx in range(num_queries):
            pivot_indices_for_query = I_faiss[query_idx]
            similarities_for_query = D_faiss[query_idx]
            valid_mask = pivot_indices_for_query != -1
            S_batch_pivot[query_idx, pivot_indices_for_query[valid_mask]] = similarities_for_query[valid_mask]
        return S_batch_pivot

    def retrieve(self, batch_test_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weighted_avg_scores_batch, _ = self.get_all_model_scores_and_similarities(
            batch_test_embeddings, return_wa_scores=True
        )
        ranked_wa_scores, ranked_indices_of_models = torch.sort(weighted_avg_scores_batch, dim=1, descending=True)
        ranked_model_ids = ranked_indices_of_models
        return ranked_model_ids, ranked_wa_scores

    def get_all_model_scores_and_similarities(
        self, batch_test_embeddings: torch.Tensor, return_wa_scores: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_test_embeddings_np = batch_test_embeddings.cpu().numpy().astype(np.float32)
        S_batch_pivot_np = self._get_similarity_to_pivots(batch_test_embeddings_np)

        numerator_np_T = self.ap_matrix_np @ S_batch_pivot_np.T
        numerator_scores_batch_np = numerator_np_T.T

        model_scores_to_return_np = numerator_scores_batch_np
        if return_wa_scores:
            sum_of_similarities_per_test_image = np.sum(S_batch_pivot_np, axis=1)
            denominator_reshaped = sum_of_similarities_per_test_image[:, np.newaxis]
            final_wa_model_scores_np = np.divide(
                numerator_scores_batch_np,
                denominator_reshaped,
                out=np.zeros_like(numerator_scores_batch_np),
                where=denominator_reshaped != 0,
            )
            model_scores_to_return_np = final_wa_model_scores_np

        model_scores_batch_torch = torch.from_numpy(model_scores_to_return_np).to(batch_test_embeddings.device)
        S_batch_pivot_torch = torch.from_numpy(S_batch_pivot_np).to(batch_test_embeddings.device)
        return model_scores_batch_torch, S_batch_pivot_torch


class FixedRankBaselineRetriever:
    def __init__(self, train_df: pd.DataFrame):
        """
        Initializes the FixedRankBaselineRetriever with a DataFrame containing training data.

        Args:
            train_df (pd.DataFrame): DataFrame containing training data with 'image_id' and 'model_id' columns.
        """
        self.train_df = train_df
        self.fixed_scores = self._get_fixed_scores(train_df)

    def _get_fixed_scores(self, train_df: pd.DataFrame):
        df_group_by_model = train_df.groupby("Model ID")["AP Rank"].mean().sort_index()
        return df_group_by_model.values * -1
