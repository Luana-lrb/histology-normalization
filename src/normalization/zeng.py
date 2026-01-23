import tensorflow as tf # foi executado fora do Vscode para roder em python 3.6 e tensorflow 1.15
import numpy as np
from base import BaseNormalizer
from utils import load_image, process_all_categories
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_color_deconvolution.stain_normalizer import StainNormalizer

class Zeng(BaseNormalizer):
    def _fit(self, reference_image):
        ref = load_image(reference_image)
        ref_batch = np.expand_dims(ref, axis=0)
        
        self.normalizer = StainNormalizer()
        self.normalizer.fit(ref_batch)

    def __call__(self, image_path):
        img = load_image(image_path)
        img_batch = np.expand_dims(img, axis=0)
        
        norm = self.normalizer.transform(img_batch)
        norm = norm[0].astype("uint8")
        return norm


if __name__ == "__main__":
    """
    IMPORTANTE:
    - Normalização Zeng deve ser executada separadamente
      usando o ambiente conda 'zeng' (TensorFlow 1.x).
    """
      
    # normalizer = Zeng("data/reference/ref1.tif")

    # process_all_categories(
    #     raw_dir="data/mini_raw",
    #     processed_dir="data/mini_processed/mini_zeng",
    #     normalizer_fn=normalizer,
    #     categories=["healthy", "mild", "moderate", "severe"]
    # )
    RAW_DIR = "data/raw"
    REFERENCE_IMAGES = {
        "ref1": "data/reference/ref1.tif",
        "ref2": "data/reference/ref2.png",
        "ref3": "data/reference/ref3.jpg",
    }
    CATEGORIES = ["healthy", "mild", "moderate", "severe"]
    
    for ref_name, ref_path in REFERENCE_IMAGES.items():
        normalizer = Zeng(ref_path)
        output_dir = f"data/processed/{ref_name}/zeng"
        print(f"\nRodando Zeng normalização com {ref_name}...")
        process_all_categories(
            raw_dir=RAW_DIR,
            processed_dir=output_dir,
            normalizer_fn=normalizer,
            categories=CATEGORIES
        )
