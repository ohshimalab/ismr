import numpy as np


class Evaluator:
    """
    Evaluator for ranking metrics such as nDCG, max_ap_relative@k, and MSE for model predictions.

    This class provides methods to evaluate predicted model scores against ground truth
    relevance labels for a set of images.

    Attributes:
        gt_relevance_matrix (np.ndarray): 2D array of ground truth relevance scores (images x models).
        image_id_to_idx (dict): Mapping from image ID to row index in gt_relevance_matrix.
        num_models (int): Number of models (columns in gt_relevance_matrix).
    """

    def __init__(self, ground_truth_relevance_matrix: np.ndarray, image_ids: list[any]):
        """
        Initialize the Evaluator.

        Args:
            ground_truth_relevance_matrix (np.ndarray): 2D array of shape (num_images, num_models)
                containing ground truth relevance scores.
            image_ids (list): List of image IDs corresponding to the rows of the relevance matrix.

        Raises:
            ValueError: If the input matrix is not 2D or image_ids length does not match matrix rows.
        """
        if not isinstance(ground_truth_relevance_matrix, np.ndarray) or ground_truth_relevance_matrix.ndim != 2:
            raise ValueError("ground_truth_relevance_matrix must be a 2D NumPy array.")
        if len(image_ids) != ground_truth_relevance_matrix.shape[0]:
            raise ValueError("Length of image_ids must match rows in ground_truth_relevance_matrix.")

        self.gt_relevance_matrix = ground_truth_relevance_matrix
        self.image_id_to_idx = {image_id: i for i, image_id in enumerate(image_ids)}
        self.num_models = self.gt_relevance_matrix.shape[1]

    def _get_gt_relevances(self, image_id: any) -> np.ndarray:
        """
        Retrieve ground truth relevances for a given image ID.

        Args:
            image_id (any): The image ID to look up.

        Returns:
            np.ndarray: Array of relevance scores for the image.

        Raises:
            KeyError: If the image ID is not found.
        """
        if image_id not in self.image_id_to_idx:
            raise KeyError(f"Image ID '{image_id}' not found in Evaluator.")
        return self.gt_relevance_matrix[self.image_id_to_idx[image_id], :]

    def evaluate(
        self,
        image_id: any,
        predicted_model_scores: list[float],
        ap_relative: list[float] = None,
        ap_score: list[float] = None,
        pred_ap_score: list[float] = None,
    ) -> dict[str, float]:
        """
        Evaluate predicted model scores for a given image using nDCG, max_ap_relative@k, and MSE metrics.

        Args:
            image_id (any): The image ID to evaluate.
            predicted_model_scores (list[float]): List of predicted scores for each model.
            ap_relative (list[float], optional): List of AP_relative values for each model (same order as scores).
            ap_score (list[float], optional): Real AP scores for each model (same order as scores).
            pred_ap_score (list[float], optional): Predicted AP scores for each model (same order as scores).

        Returns:
            dict[str, float]: Dictionary with nDCG@k, max_ap_relative@k, and mse_ap_score metrics.

        Raises:
            ValueError: If the length of predicted_model_scores does not match number of models.
        """
        from .utils import get_ndcg

        if len(predicted_model_scores) != self.num_models:
            raise ValueError(
                f"Length of predicted_model_scores ({len(predicted_model_scores)}) "
                f"must match num_models ({self.num_models})."
            )

        gt_relevances_for_image = self._get_gt_relevances(image_id)
        ranked_model_indices = np.argsort(predicted_model_scores)[::-1]
        relevances_in_predicted_order = gt_relevances_for_image[ranked_model_indices]

        metrics = {}
        for k_val in [3, 5]:
            metrics[f"ndcg@{k_val}"] = get_ndcg(relevances_in_predicted_order.tolist(), k=k_val, boost=True)

        # Calculate max_ap_relative@k if ap_relative is provided
        if ap_relative is not None:
            ap_relative = np.array(ap_relative)
            ap_relative_in_predicted_order = ap_relative[ranked_model_indices]
            for k_val in [3, 5]:
                metrics[f"max_ap_relative@{k_val}"] = float(np.max(ap_relative_in_predicted_order[:k_val]))

        # Calculate MSE between real ap_score and predicted ap_score if both are provided
        if ap_score is not None and pred_ap_score is not None:
            ap_score = np.array(ap_score)
            pred_ap_score = np.array(pred_ap_score)
            if ap_score.shape != pred_ap_score.shape:
                raise ValueError("ap_score and pred_ap_score must have the same shape.")
            metrics["mse_ap_score"] = float(np.mean((ap_score - pred_ap_score) ** 2))

        return metrics
