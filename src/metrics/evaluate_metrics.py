import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from metrics import metricas


CLASSES = ["healthy", "mild", "moderate", "severe"]

METHODS = [
    "histogram_matching",
    "macenko",
    "modified_reinhard",
    "reinhard",
    "vahadane",
    "zeng"
]

REFERENCIAS = {
    "ref1": "data/reference/ref1.tif",
    "ref2": "data/reference/ref2.png",
    "ref3": "data/reference/ref3.jpg"
}

results = []

# Método com referência individual
for ref_name, ref_path in REFERENCIAS.items():

    print(f"\nProcessando a referência: {ref_name}")

    ref_img = cv2.imread(ref_path)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ref_img = ref_img.astype(np.uint8)

    for method in METHODS:

        print(f"\n Processando o método: {method}")

        for classe in CLASSES:

            norm_dir = f"data/processed/{ref_name}/{method}/{classe}"
            raw_dir = f"data/raw/{classe}"

            if not os.path.exists(norm_dir):
                print(f"Pasta não encontrada: {norm_dir}")
                continue

            images = os.listdir(norm_dir)

            for img_name in tqdm(
                images,
                desc=f"{method} | {classe}"
            ):

                norm_path = os.path.join(norm_dir, img_name)
                raw_path = os.path.join(raw_dir, img_name)

                norm_img = cv2.imread(norm_path)
                raw_img = cv2.imread(raw_path)

                if norm_img is None or raw_img is None:
                    continue
                
                norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                
                norm_img = norm_img.astype(np.uint8)
                raw_img = raw_img.astype(np.uint8)
                
                if norm_img.shape != raw_img.shape:

                    norm_img = cv2.resize(
                        norm_img,
                        (raw_img.shape[1], raw_img.shape[0])
                    )
                    
                # Métricas com ORIGINAL
                met_orig = metricas(norm_img, raw_img)
                
                # Métricas com REFERÊNCIA
                ref_img_resized = ref_img
                if norm_img.shape != ref_img_resized.shape:

                    ref_img_resized = cv2.resize(
                        ref_img_resized,
                        (norm_img.shape[1], norm_img.shape[0])
                    )

                met_ref = metricas(norm_img, ref_img_resized)

                results.append({

                    "image": img_name,
                    "class": classe,
                    "method": method,
                    "reference": ref_name,

                    # ORIGINAL
                    "SSIM_original": met_orig["SSIM"],
                    "QSSIM_original": met_orig["QSSIM"],
                    "PSNR_original": met_orig["PSNR"],
                    "PCC_original": met_orig["PCC"],
                    "DeltaE_original": met_orig["DeltaE"],

                    # REFERÊNCIA
                    "SSIM_reference": met_ref["SSIM"],
                    "QSSIM_reference": met_ref["QSSIM"],
                    "PSNR_reference": met_ref["PSNR"],
                    "PCC_reference": met_ref["PCC"],
                    "DeltaE_reference": met_ref["DeltaE"]
                })

        print(f"Finalizado: {method}")

        # salva parcial
        df_partial = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        df_partial.to_csv(
            "results/metrics_partial.csv",
            index=False
        )

# Multitarget Macenko
print("\nProcessando multitarget_macenko")

for classe in CLASSES:

    norm_dir = f"data/processed/multitarget_macenko/{classe}"
    raw_dir = f"data/raw/{classe}"

    images = os.listdir(norm_dir)

    for img_name in tqdm(
        images,
        desc=f"multitarget | {classe}"
    ):

        norm_path = os.path.join(norm_dir, img_name)
        raw_path = os.path.join(raw_dir, img_name)

        norm_img = cv2.imread(norm_path)
        raw_img = cv2.imread(raw_path)
        
        if norm_img is None or raw_img is None:
            continue
        
        norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        norm_img = norm_img.astype(np.uint8)
        raw_img = raw_img.astype(np.uint8)


        if norm_img.shape != raw_img.shape:

            norm_img = cv2.resize(
                norm_img,
                (raw_img.shape[1], raw_img.shape[0])
            )

        met_orig = metricas(norm_img, raw_img)

        results.append({

            "image": img_name,
            "class": classe,
            "method": "multitarget_macenko",
            "reference": "multi",

            "SSIM_original": met_orig["SSIM"],
            "QSSIM_original": met_orig["QSSIM"],
            "PSNR_original": met_orig["PSNR"],
            "PCC_original": met_orig["PCC"],
            "DeltaE_original": met_orig["DeltaE"]
        })

# CSV final
df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
df.to_csv("results/metrics.csv", index=False)

summary = df.groupby(
    ["method", "reference"]
).mean(numeric_only=True)

summary.to_csv("results/metrics_summary.csv")

print("\nFINALIZADO COM SUCESSO!")