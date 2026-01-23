from skimage.exposure import match_histograms
from base import BaseNormalizer
from utils import load_image, process_all_categories
class HistogramMatching(BaseNormalizer):
    def _fit(self, reference_image):
        self.reference_image = load_image(reference_image)

    def __call__(self, image_path):
        img = load_image(image_path)
        matched = match_histograms(img, self.reference_image, channel_axis=-1)
        matched = matched.clip(0, 255).astype("uint8")
        return matched
    
if __name__ == "__main__":
    normalizer = HistogramMatching("data/reference/ref1.tif")

    process_all_categories(
        raw_dir="data/mini_raw",
        processed_dir="data/mini_processed/mini_histogram_matching",
        normalizer_fn=normalizer,
        categories=["healthy", "mild", "moderate", "severe"]
    )