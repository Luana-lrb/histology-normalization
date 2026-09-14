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


EXPERIMENT_1 = [
    ("raw", None, None)
]


for normalization in NORMALIZATIONS:
    EXPERIMENT_1.append(
        ("processed", "ref1", normalization)
    )


def create_experiment_2(best_normalizations):
    experiments = [
        ("raw", None, None)
    ]

    for normalization in best_normalizations:
        for reference in REFERENCES:
            experiments.append(
                ("processed", reference, normalization)
            )

    experiments.append(
        ("processed", None, "multitarget_macenko")
    )

    return experiments