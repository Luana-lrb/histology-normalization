from pathlib import Path
from typing import Optional, Callable
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from config import CLASS_NAMES

CLASS_TO_IDX = {
    class_name: idx
    for idx, class_name in enumerate(CLASS_NAMES)
}

class HistologyDataset(Dataset):
    def __init__(
        self,
        csv_file: Path,
        root_dir: Path,
        transform: Optional[Callable] = None
    ):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = self.root_dir / row["image"]
        
        if not image_path.exists(): 
            raise FileNotFoundError(
                f"Imagem não encontrada: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        label = CLASS_TO_IDX[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label