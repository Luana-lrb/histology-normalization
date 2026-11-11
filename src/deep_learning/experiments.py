NORMALIZATIONS = [
    "macenko",
    "reinhard",
    "modified_reinhard",
    "vahadane",
    "zeng",
    "histogram_matching"
]


REFERENCES = [
    "ref1",
    "ref2",
    "ref3"
]

EXPERIMENT_0 = [
    ("with_augmentation", True),
    ("without_augmentation", False)
]

EXPERIMENT_1 = []


for normalization in NORMALIZATIONS:
    EXPERIMENT_1.append(
        ("processed", "ref1", normalization)
    )


def create_experiment_2(best_normalizations):
    experiments = []

    # referência 1 já foi utilizada no Experimento 1
    # treinamos apenas com as referências 2 e 3
    for normalization in best_normalizations:
        for reference in ["ref2", "ref3"]:
            experiments.append(
                ("processed", reference, normalization)
            )
    experiments.append(
        ("processed", None, "multitarget_macenko")
    )

    return experiments
