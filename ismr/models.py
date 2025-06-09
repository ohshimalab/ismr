import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import ViTModel

from . import utils

current_dir = Path(__file__).parent.resolve()
allRank_dir = current_dir.parent / "allRank"

sys.path.insert(1, (allRank_dir / "allrank").as_posix())
sys.path.append(allRank_dir.as_posix())
sys.path.append((allRank_dir / "allrank" / "models" / "losses").as_posix())


try:
    from neuralNDCG import neuralNDCG
except ImportError:
    raise ImportError(
        "Could not import neuralNDCG. " "Make sure allRank is cloned in the project root and the path is correct."
    )

DEFAULT_ALPHA = 0.5
DEFAULT_NEURALNDCG_K = 1
DEFAULT_MODEL_DIM = 768
DEFAULT_IMAGE_DIM = 768
DEFAULT_HIDDEN_DIM = 512


class APPredictor(pl.LightningModule):
    """APPredictor"""

    def __init__(
        self,
        vit: ViTModel,
        model_numbers: int,
        lr: float,
        alpha: float = DEFAULT_ALPHA,
        neuralNDCG_k: int = DEFAULT_NEURALNDCG_K,
        model_dim: int = DEFAULT_MODEL_DIM,
        image_dim: int = DEFAULT_IMAGE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        model_vectors=None,
    ):
        """
        Initialize the APPredictor model.

        Args:
            vit (ViTModel): Pre-trained ViT model for image feature extraction.
            model_numbers (int): Number of target models.
            lr (float): Learning rate for the optimizer.
            alpha (float, optional): Weight for the MSE loss.
            neuralNDCG_k (int, optional): k value for neuralNDCG loss.
            model_dim (int, optional): Dimension of the model vectors.
            image_dim (int, optional): Dimension of the image vectors.
            hidden_dim (int, optional): Dimension of the hidden layer in the MLP.
            model_vectors (torch.Tensor, optional): Pre-initialized model vectors.
        """

        super().__init__()
        self.save_hyperparameters(ignore=["vit"])
        self.vit = vit
        self.model_dim = model_dim
        self.model_numbers = model_numbers
        self.lr = lr
        self.alpha = alpha
        self.neuralNDCG_k = neuralNDCG_k

        self.model_vectors = torch.nn.Parameter(
            model_vectors if model_vectors is not None else torch.rand(model_numbers, model_dim)
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(image_dim + model_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        )

        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_l2r = neuralNDCG

        # Tracking losses and metrics
        self._init_metrics()

    def _init_metrics(self):
        self.train_epoch_losses = {"mse": [], "l2r": []}
        self.val_epoch_losses = {"mse": [], "l2r": []}
        self.train_batch_losses = {"mse": [], "l2r": []}
        self.val_batch_losses = {"mse": [], "l2r": []}
        self.train_batch_ndcg = {k: [] for k in [3, 5]}
        self.val_batch_ndcg = {k: [] for k in [3, 5]}
        self.train_epoch_ndcg = {k: [] for k in [3, 5]}
        self.val_epoch_ndcg = {k: [] for k in [3, 5]}

    def forward(self, images):
        image_vectors = self.vit(pixel_values=images).last_hidden_state[:, 0, :]

        image_vectors = image_vectors.unsqueeze(1).repeat(1, self.model_numbers, 1)
        model_vectors = self.model_vectors.unsqueeze(0).repeat(image_vectors.shape[0], 1, 1)
        z = torch.cat((image_vectors, model_vectors), dim=2)
        z = self.mlp(z).squeeze(2)
        return z

    def _calculate_loss(self, preds, targets):
        mse_loss = self.alpha * self.criterion_mse(preds, targets)
        l2r_loss = (1 - self.alpha) * self.criterion_l2r(preds, targets, k=self.neuralNDCG_k)
        return mse_loss, l2r_loss

    def _calculate_ndcg(self, preds, relevances):
        preds_np = preds.detach().cpu().numpy()
        relevances_np = relevances.cpu().numpy()
        sorted_indices = np.argsort(preds_np, axis=1)[:, ::-1]
        sorted_relevances = np.take_along_axis(relevances_np, sorted_indices, axis=1)
        ndcgs = {k: [utils.get_ndcg(row.tolist(), k, True) for row in sorted_relevances] for k in [3, 5]}
        return tuple(np.mean(ndcgs[k]) for k in [3, 5])

    def _shared_step(self, batch, prefix):
        _, _, img, model_scores, model_relative_aps, relevances = batch
        z = self.forward(img)
        mse_loss, l2r_loss = self._calculate_loss(z, model_scores)
        loss = mse_loss + l2r_loss
        ndcg_3, ndcg_5 = self._calculate_ndcg(z, relevances)
        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{prefix}_ndcg_3", ndcg_3, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{prefix}_ndcg_5", ndcg_5, on_epoch=True, prog_bar=True, logger=True)
        batch_losses = getattr(self, f"{prefix}_batch_losses")
        batch_ndcg = getattr(self, f"{prefix}_batch_ndcg")
        batch_losses["mse"].append(mse_loss.item())
        batch_losses["l2r"].append(l2r_loss.item())
        batch_ndcg[3].append(ndcg_3)
        batch_ndcg[5].append(ndcg_5)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    def _epoch_end(self, prefix):
        batch_losses = getattr(self, f"{prefix}_batch_losses")
        batch_ndcg = getattr(self, f"{prefix}_batch_ndcg")
        epoch_losses = getattr(self, f"{prefix}_epoch_losses")
        epoch_ndcg = getattr(self, f"{prefix}_epoch_ndcg")
        avg_mse_loss = np.mean(batch_losses["mse"]) if batch_losses["mse"] else 0.0
        avg_l2r_loss = np.mean(batch_losses["l2r"]) if batch_losses["l2r"] else 0.0
        avg_ndcg = {k: np.mean(batch_ndcg[k]) if batch_ndcg[k] else 0.0 for k in [3, 5]}
        epoch_losses["mse"].append(avg_mse_loss)
        epoch_losses["l2r"].append(avg_l2r_loss)
        for k in [3, 5]:
            epoch_ndcg[k].append(avg_ndcg[k])
        setattr(self, f"{prefix}_batch_losses", {"mse": [], "l2r": []})
        setattr(self, f"{prefix}_batch_ndcg", {k: [] for k in [3, 5]})

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)


class Retriever(ABC):
    """
    Abstract base class for a retriever.
    A retriever takes a batch of inputs and returns ranked model indices and their scores.
    """

    @abstractmethod
    def retrieve(self, batch_input: any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves and ranks models for a given batch of inputs.

        Args:
            batch_input (Any): The input batch for the retriever (e.g., image tensors).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - ranked_model_indices (torch.Tensor): Shape B x M (Batch size, Num ranked models).
                  Contains model indices (0 to M-1), ranked by preference.
                - ranked_model_scores (torch.Tensor): Shape B x M. Scores for ranked models.
        """
        pass


class APPredictorRetriever(Retriever):
    """
    A Retriever that uses a trained APPredictor model to rank models for given images.
    """

    def __init__(self, appredictor_model: APPredictor):
        """
        Initializes the APPredictorRetriever.

        Args:
            appredictor_model (APPredictor): A trained instance of the APPredictor model.
            model_names (List[str]): A list of M model string names, corresponding to the
                                     order of outputs from APPredictor.
        """
        super().__init__()
        if not isinstance(appredictor_model, APPredictor):
            print(f"Warning: appredictor_model is type {type(appredictor_model)}, expected APPredictor.")
        if not hasattr(appredictor_model, "forward") or not callable(appredictor_model.forward):
            raise ValueError("appredictor_model must be a valid model with a forward method.")

        self.model = appredictor_model
        self.model.eval()
        self.num_models = len(appredictor_model.model_vectors)
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:  # Handle case where model might have no parameters (unlikely for APPredictor)
            print("Warning: Could not determine model device, defaulting to CPU.")
            self.device = torch.device("cpu")

        # Internal representation of model IDs will be their indices (0 to M-1)
        self.model_indices_tensor = torch.arange(self.num_models, dtype=torch.long, device=self.device)
        print(f"APPredictorRetriever initialized with {self.num_models} models. Device: {self.device}")

    @torch.no_grad()
    def retrieve(self, batch_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts model scores for a batch of images using the APPredictor and ranks them.

        Args:
            batch_images (torch.Tensor): Shape B x C x H x W, suitable for APPredictor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ranked_model_indices (torch.Tensor): B x N_models, integer indices (0 to M-1)
                                                       of models, ranked by predicted score.
                - ranked_model_scores (torch.Tensor): B x N_models, predicted scores.
        """
        if not isinstance(batch_images, torch.Tensor):
            raise TypeError("batch_images must be a torch.Tensor.")

        batch_images = batch_images.to(self.device)
        all_model_scores = self.model(batch_images)  # Expected shape: B x M (num_models)

        # Sort scores in descending order to get ranked scores and their original indices
        ranked_scores, ranked_indices = torch.sort(all_model_scores, dim=1, descending=True)

        # ranked_indices are already the 0-indexed model IDs based on their score ranking
        ranked_model_indices = ranked_indices

        return ranked_model_indices, ranked_scores
