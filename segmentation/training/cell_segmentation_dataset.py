import numpy as np

from torch.utils.data import Dataset


class ConicDataset(Dataset):
    """ Pytorch data set for CoNIC Challenge """

    def __init__(self, root_dir, mode, transform=lambda x: x):
        """

        :param root_dir: Directory containing the dataset.
            :type root_dir: pathlib Path object.
        :param mode: 'train' or 'val'.
            :type mode: str
        :param transform: transforms.
            :type transform:
        :return: Dict (image, cell_label, border_label, id).
        """
        
        if mode == 'train':
            self.imgs = np.load(root_dir / "train_images.npy")
            self.labels = np.load(root_dir / "train_labels.npy")
            # Add some randomness
            ids = np.arange(len(self.imgs))
            np.random.shuffle(ids)
            self.imgs = self.imgs[ids]
            self.labels = self.labels[ids]
            assert self.imgs.shape[0] == self.labels.shape[0], "Missmatch between images.npy and labels_train.npy"
        elif mode == 'val':
            self.imgs = np.load(root_dir / "valid_images.npy")
            self.labels = np.load(root_dir / "valid_labels.npy")
            assert self.imgs.shape[0] == self.labels.shape[0], "Missmatch between images.npy and labels_train.npy"
        elif mode == 'eval':
            self.imgs = np.load(root_dir / "valid_images.npy")
            self.labels = np.load(root_dir / "valid_gts.npy").astype(np.int64)  # pytorchs default_colate cannot handle uint16

        self.root_dir = root_dir
        self.mode = mode
        self.len = len(self.imgs)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {'image': np.copy(self.imgs[idx, ...]),
                  'label': np.copy(self.labels[idx, ...]),
                  'id': idx}
        sample = self.transform(sample)
        return sample
