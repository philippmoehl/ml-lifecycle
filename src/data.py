import cv2
from torch.utils.data import Dataset


class LeafDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for leaf_path in self.data_path.iterdir():
            image_paths.extend(leaf_path.glob('*.[jJ][pP][gG]'))
        return image_paths

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

        leaf_class = image_path.parent.name
        return image, leaf_class
