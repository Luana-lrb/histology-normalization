from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATASET_DIR,
    SPLITS_DIR,
    SPLIT_SEED,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    TEST_SIZE
)

def extract_group_id(image_path: Path, label: str) -> str:
    """
    Identifica a imagem de origem de cada ROI.
    """

    source_name = re.sub(
        r"-roi\d+$",
        "",
        image_path.stem
    )

    return f"{label}/{source_name}"


def build_dataframe() -> pd.DataFrame:
    """
    Percorre o dataset RAW e cria um DataFrame contendo
    o caminho relativo da imagem e sua respectiva classe
    """

    data = []

    for class_dir in sorted(RAW_DATASET_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for image_path in sorted(class_dir.iterdir()):

            if image_path.is_file():

                relative_path = image_path.relative_to(RAW_DATASET_DIR)

                data.append({
                    "image": relative_path.as_posix(),
                    "label": label,
                    "group": extract_group_id(image_path, label)
                })
    print(f"Total de imagens encontradas: {len(data)}")
    print(pd.DataFrame(data).head())

    return pd.DataFrame(data)


def generate_splits():
    """
    Gera os conjuntos de treino, validação e teste
    """

    df = build_dataframe()
    
    groups_df = ( # separando imagens e não ROIs
        df[["group", "label"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    train_groups, temp_groups = train_test_split(
        groups_df,
        train_size=TRAIN_SIZE,
        stratify=groups_df["label"], # para as classes serem balanceadas entre os conjuntos
        random_state=SPLIT_SEED
    )

    validation_ratio = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)

    validation_groups, test_groups = train_test_split(
        temp_groups,
        train_size=validation_ratio,
        stratify=temp_groups["label"],
        random_state=SPLIT_SEED
    )
    
    train_df = df[
        df["group"].isin(train_groups["group"])
    ].copy()

    validation_df = df[
        df["group"].isin(validation_groups["group"])
    ].copy()

    test_df = df[
        df["group"].isin(test_groups["group"])
    ].copy()
    
    train_group_set = set(train_df["group"])
    validation_group_set = set(validation_df["group"])
    test_group_set = set(test_df["group"])

    assert train_group_set.isdisjoint(validation_group_set), (
        "Existem imagens de origem repetidas entre treino e validação."
    )

    assert train_group_set.isdisjoint(test_group_set), (
        "Existem imagens de origem repetidas entre treino e teste."
    )

    assert validation_group_set.isdisjoint(test_group_set), (
        "Existem imagens de origem repetidas entre validação e teste."
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    validation_df.to_csv(SPLITS_DIR / "validation.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    print("Splits gerados com sucesso!\n")

    print(f"Treino:     {len(train_df)} imagens")
    print(f"Validação:  {len(validation_df)} imagens")
    print(f"Teste:      {len(test_df)} imagens")

    print_distribution(train_df, "Treino")
    print_distribution(validation_df, "Validação")
    print_distribution(test_df, "Teste")

def print_distribution(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name}")
    print(df["label"].value_counts().sort_index())

if __name__ == "__main__":
    generate_splits()
