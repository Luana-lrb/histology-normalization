from torchvision import transforms
from base import BaseNormalizer
from utils import load_image, tensor_to_uint8, process_all_categories
from torchstain.torch.normalizers.reinhard import TorchReinhardNormalizer

class ModifiedReinhard(BaseNormalizer):
    def _fit(self, reference_image):
        target = load_image(reference_image)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])

        self.normalizer = TorchReinhardNormalizer(method="modified")
        self.normalizer.fit(self.transform(target))

    def __call__(self, image_path):
        img = load_image(image_path)
        img_tensor = self.transform(img)

        norm = self.normalizer.normalize(img_tensor)
        return tensor_to_uint8(norm)


if __name__ == "__main__":
    normalizer = ModifiedReinhard("data/reference/ref1.tif")
    process_all_categories(
        raw_dir="data/mini_raw",
        processed_dir="data/processed/mini_modified_reinhard",
        normalizer_fn=normalizer,
        categories=["healthy", "mild", "moderate", "severe"]
    )
