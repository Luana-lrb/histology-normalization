from torchvision import transforms
from base import BaseNormalizer
from utils import load_image, tensor_to_uint8, process_all_categories
import torchstain


class MultiTargetMacenko(BaseNormalizer):
    def __init__(self, reference_images, norm_mode="avg-post"):
        self.norm_mode = norm_mode
        super().__init__(reference_images)

    def _fit(self, reference_images):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])

        refs = [
            transform(load_image(img_path))
            for img_path in reference_images
        ]

        self.normalizer = torchstain.normalizers.MultiMacenkoNormalizer(norm_mode=self.norm_mode)
        self.normalizer.fit(refs)

    def __call__(self, image_path):
        img = load_image(image_path)
        img_tensor = transforms.ToTensor()(img) * 255

        norm, _, _ = self.normalizer.normalize(img_tensor, stains=True)
        return tensor_to_uint8(norm)

if __name__ == "__main__":
    refs = [
        "data/reference/ref1.tif",
        "data/reference/ref2.png",
        "data/reference/ref3.jpg",
    ]

    normalizer = MultiTargetMacenko(
        reference_images=refs,
        norm_mode="avg-post"
    )

    process_all_categories(
        raw_dir="data/mini_raw",
        processed_dir="data/processed/mini_multitarget_macenko",
        normalizer_fn=normalizer,
        categories=["healthy", "mild", "moderate", "severe"]
    )
