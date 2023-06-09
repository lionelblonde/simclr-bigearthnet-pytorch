import math

from torch.utils.data import DataLoader

from helpers.dataloader_utils.bigearthnet_utils.dataset import BigEarthNetDataset


class BigEarthNetDataloader(DataLoader):

    def __init__(self, dataset: BigEarthNetDataset, batch_size: int, *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.dataset_length = len(dataset)

    def __len__(self):
        """Overwrite because relays the method of the `Dataset` class otherwise"""
        if self.batch_size is None:
            raise ValueError(f"invalid batch size ({self.batch_size}); can't be None!")
        return math.ceil(self.dataset_length // self.batch_size)


def get_dataloader(
    *,
    dataset_handle: str,
    data_path: str,
    split_path: str,
    batch_size: int,
    train_stage: bool = False,
    val_stage: bool = False,
    test_or_inference_stage: bool = False,
    num_transforms: int = 2,
    with_labels: bool = False,
    memory: bool = False,
    shuffle: bool = True,
):

    if dataset_handle == 'bigearthnet':
        dataloader = BigEarthNetDataloader(
            BigEarthNetDataset(
                data_path=data_path,
                split_path=split_path,
                image_size=120,
                train_stage=train_stage,
                val_stage=val_stage,
                test_or_inference_stage=test_or_inference_stage,
                num_transforms=num_transforms,
                bands=BigEarthNetDataset.all_bands(),
                with_labels=with_labels,
                memory=memory,
            ),
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=True,
        )
        return dataloader
    else:
        raise ValueError(f"{dataset_handle} is not a valid dataset name.")
