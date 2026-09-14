from torchvision import transforms
from PIL import ImageOps
from config import IMAGE_SIZE, USE_AUGMENTATION

def pad_to_square(image):
    """
    Completa a imagem com bordas brancas até que ela fique quadrada,
    sem alterar sua proporção.
    """

    width, height = image.size
    max_side = max(width, height)

    horizontal_padding = max_side - width
    vertical_padding = max_side - height

    left = horizontal_padding // 2
    right = horizontal_padding - left

    top = vertical_padding // 2
    bottom = vertical_padding - top

    return ImageOps.expand(
        image,
        border=(left, top, right, bottom),
        fill=(255, 255, 255)
    )

# Estatísticas do ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


if USE_AUGMENTATION:

    train_transform = transforms.Compose([
        transforms.Lambda(pad_to_square),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=10,
            fill=(255, 255, 255)
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=(255, 255, 255)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

else:

    train_transform = transforms.Compose([
        transforms.Lambda(pad_to_square),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


validation_transform = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


test_transform = validation_transform