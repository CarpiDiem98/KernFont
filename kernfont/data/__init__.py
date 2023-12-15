from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from kernfont.data.dataset import Font


def get_dataloader(annotation_file: str, **kwargs):
    """Get dataloader for training and validation

    Args:
        annotations_file (str): Path to annotations file

    Returns:
        torch.utils.data.DataLoader, torch.utils.data.DataLoader: Train and validation dataloaders
    """
    dataset = get_dataset(annotation_file)
    train_dataset, validation_dataset = __splitter(dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=kwargs.get("batch_size"),
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=kwargs.get("batch_size"),
        shuffle=False,
    )
    return train_loader, validation_loader


def __splitter(dataset, percentage=0.8):
    """Split dataset into train and validation sets

    Args:
        dataset (torch.utils.data.Dataset): Dataset to split

    Returns:
        torch.utils.data.Dataset, torch.utils.data.Dataset: Train and validation sets
    """
    train_size = int(percentage * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
    )
    return train_dataset, validation_dataset


def get_dataset(annotation_file: str):
    """Get dataset from annotations file

    Args:
        annotation_file (str): Path to annotations file

    Returns:
        torch.utils.data.Dataset: Dataset
    """
    dataset = Font(
        annotation_file,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((1000, 1000), antialias=True),
            ]
        ),
    )
    return dataset
