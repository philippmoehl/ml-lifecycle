import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class LeafDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        self.initialize()

    def initialize(self):
        label_to_int = {
            label: idx for idx, label in enumerate(set(self.labels))}
        self.labels = [label_to_int[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path.as_posix())
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image = image)['image']

        label = torch.tensor(self.labels[idx]).long()
        return image, label


class WrappedDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for x, y in self.dl:
            yield (x.to(self.device), y.to(self.device))


def get_image_labels(data_path):
    image_paths = []
    labels = []

    for label_path in data_path.iterdir():
        label_image_paths = sorted(label_path.glob('*.[jJp][pPn][gGg]'))
        image_paths.extend(label_image_paths)
        labels.extend(len(label_image_paths) * [label_path.name])

    return image_paths, labels


def image_labels_split(data_path, test_size, seed):
    X, y = get_image_labels(data_path)
    return  train_test_split(X, y, test_size=test_size, random_state=seed)
