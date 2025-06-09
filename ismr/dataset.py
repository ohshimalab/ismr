import os

import torch
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset"""

    def __init__(
        self,
        root_dir: str,
        image_names: list[str],
        image_paths: list[str],
        image_categories: list[str],
        model_scores_lists: list[list[float]],
        model_relative_ap_lists: list[list[float]],
        model_relevance_lists: list[list[float]],
        transform=None,
    ):
        """Initialize the dataset.

        Args:
            root_dir (str): path to the root directory.
            image_names (list[str]): image names.
            image_paths (list[str]): image paths.
            image_categories (list[str]): image categories.
            model_scores_lists (list[list[float]]): scores of the models for each image.
            model_relevance_lists (list[list[float]]): relevance score of the models for each image.
            transform (_type_, optional): data transforms. Defaults to None.
        """
        self.root_dir = root_dir
        self.image_names = image_names
        self.image_paths = image_paths
        self.image_categories = image_categories
        self.model_scores_lists = torch.tensor(model_scores_lists)
        self.model_relative_ap_lists = torch.tensor(model_relative_ap_lists)
        self.model_relevance_lists = torch.tensor(model_relevance_lists)
        self.transform = transform

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Get the item at the specified index."""
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return (
            self.image_categories[index],
            self.image_names[index],
            img,
            self.model_scores_lists[index],
            self.model_relative_ap_lists[index],
            self.model_relevance_lists[index],
        )
