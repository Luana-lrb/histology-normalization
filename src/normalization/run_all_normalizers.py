"""
IMPORTANTE:
- Normalização Zeng deve ser executada separadamente
  usando o ambiente conda 'zeng' (TensorFlow 1.x).
"""

from macenko import Macenko
from multitarget_macenko import MultiTargetMacenko
from reinhard import Reinhard
from modified_reinhard import ModifiedReinhard
from vahadane import Vahadane
from histogram_matching import HistogramMatching
from utils import process_all_categories


RAW_DIR = "data/raw"
REFERENCE_IMAGES = {
    "ref1": "data/reference/ref1.tif",
    "ref2": "data/reference/ref2.png",
    "ref3": "data/reference/ref3.jpg",
}
CATEGORIES = ["healthy", "mild", "moderate", "severe"]


def run_normalizer(name, normalizer, output_dir):
    print(f"\nRodando {name} normalização...")
    process_all_categories(
        raw_dir=RAW_DIR,
        processed_dir=output_dir,
        normalizer_fn=normalizer,
        categories=CATEGORIES
    )


def main():
    
    # Normalizações com target único
    for ref_name, ref_path in REFERENCE_IMAGES.items():

        normalizers = {
            "macenko": (
                Macenko(ref_path),
                f"data/processed/{ref_name}/macenko"
            ),
            "reinhard": (
                Reinhard(ref_path),
                f"data/processed/{ref_name}/reinhard"
            ),
            "modified_reinhard": (
                ModifiedReinhard(ref_path),
                f"data/processed/{ref_name}/modified_reinhard"
            ),
            "vahadane": (
                Vahadane(ref_path),
                f"data/processed/{ref_name}/vahadane"
            ),
            "histogram_matching": (
                HistogramMatching(ref_path),
                f"data/processed/{ref_name}/histogram_matching"
            )
        }

        for name, (normalizer, out_dir) in normalizers.items():
            run_normalizer(name, normalizer, out_dir)
    
    # Normalização MultiTarget Macenko
    
    multitarget_refs = [
        "data/reference/ref1.tif",
        "data/reference/ref2.png",
        "data/reference/ref3.jpg",
    ]

    multitarget_normalizer = MultiTargetMacenko(
        reference_images=multitarget_refs,
        norm_mode="avg-post"
    )

    process_all_categories(
        raw_dir=RAW_DIR,
        processed_dir="data/processed/multitarget_macenko",
        normalizer_fn=multitarget_normalizer,
        categories=CATEGORIES
    )

    print("\nTodas as normalizações concluídas com sucesso.")
    print("Lembrete: A normalização Zeng deve ser executada separadamente.")


if __name__ == "__main__":
    main()
