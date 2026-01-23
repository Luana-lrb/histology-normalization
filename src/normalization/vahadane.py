from wsi_normalizer import TorchVahadaneNormalizer  
from base import BaseNormalizer
from utils import load_image, process_all_categories

class Vahadane(BaseNormalizer):
    def _fit(self, reference_image):
        
        target = load_image(reference_image)
        self.normalizer = TorchVahadaneNormalizer(device="cpu")
        self.normalizer.fit(target)

    def __call__(self, image_path):
        img = load_image(image_path)
        norm = self.normalizer.transform(img)
        return norm


if __name__ == "__main__":
    normalizer = Vahadane("data/reference/ref1.tif")
    
    process_all_categories(
        raw_dir="data/mini_raw",
        processed_dir="data/mini_processed/mini_vahadane",
        normalizer_fn=normalizer,
        categories=["healthy","mild","moderate","severe"]
    )