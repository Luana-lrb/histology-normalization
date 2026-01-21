import os
from pathlib import Path
from PIL import Image    
from tqdm import tqdm
import cv2
import numpy as np

def load_image(image_path):
    try:
        img = Image.open(image_path)
        return np.array(img.convert("RGB"))
    except Exception:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Não foi possível carregar {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def tensor_to_uint8(img):
    # torch Tensor
    if hasattr(img, "detach"):  
        img = img.detach().cpu().numpy()

    # Caso venha como (3, H, W)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    # Caso venha como (H, W, 3)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Formato inválido da imagem normalizada: {img.shape}")

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def process_directory(input_dir, output_dir, normalize_fn):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_exts
    ]
    failed_files = []

    for img_file in image_files:
        try:
            result = normalize_fn(img_file)
            norm = result

            cv2.imwrite(
                str(output_path / img_file.name),
                cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
            )

        except Exception as e:
            print(f"Erro em {img_file.name}: {e}")
            failed_files.append(img_file.name)
    if failed_files:
        print("Arquivos que falharam na normalização:")

def process_all_categories(raw_dir, processed_dir, normalizer_fn, categories):

    for category in categories:
        input_dir = os.path.join(raw_dir, category)
        output_dir = os.path.join(processed_dir, category)
        
        if os.path.exists(input_dir):
            process_directory(input_dir, output_dir, normalizer_fn)
            print(f"Categoria {category} processada com sucesso!")
        else:
            print(f"Aviso: Diretório {input_dir} não encontrado. Pulando...")

