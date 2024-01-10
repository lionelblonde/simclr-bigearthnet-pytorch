from pathlib import Path

from helpers.dataloader_utils import path2str


def split_bigearthnet_official(num_classes: int) -> list[str]:
    where_splits = (f"./splits/BigEarthNet-S2_{num_classes}-classes_OFFICIAL/splits")
    txt_files = ['train.txt', 'val.txt', 'test.txt']
    return [path2str(Path(where_splits).joinpath(f)) for f in txt_files]


def split_dataset(dataset_handle: str, num_classes: int) -> list[str]:
    paths_list = []

    if dataset_handle == "bigearthnet":
        paths_list = split_bigearthnet_official(num_classes)

    return paths_list
