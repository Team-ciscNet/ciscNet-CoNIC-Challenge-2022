import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class ConicDataset(Dataset):
    """ Pytorch data set for CoNIC Challenge """

    def __init__(self, root_dir, mode='train', transform=lambda x: x, train_split=80):
        """

        :param root_dir: Directory containing the dataset.
            :type root_dir: pathlib Path object.
        :param mode: 'train' or 'val'.
            :type mode: str
        :param transform: transforms.
            :type transform:
        :param train_split: percent of the data used for training
            :type train_split: int
        :return: Dict (image, cell_label, border_label, id).
        """

        imgs = np.load(root_dir/"images.npy")
        
        if mode in ['train', 'val']:
            labels = np.load(root_dir / "labels.npy")
            assert imgs.shape[0] == labels.shape[0], "Missmatch between images.npy and labels_train.npy"
            counts = pd.read_csv(root_dir / "counts.csv")
        elif mode == 'eval':
            labels = np.load(root_dir / "gts.npy").astype(np.int64)  # pytorchs default_colate cannot handle uint16
            counts = pd.read_csv(root_dir / "counts.csv")

        self.root_dir = root_dir
        self.mode = mode
        self.train_split = train_split
        self.ids = self.extract_train_val_ids(imgs.shape[0], 0)
        self.imgs = imgs[self.ids, ...]
        self.len = len(self.ids)
        if mode in ['train', 'val', 'eval']:
            self.labels = labels[self.ids, ...]
            self.counts = self.get_counts(counts=counts)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {'image': np.copy(self.imgs[idx, ...]),
                  'label': np.copy(self.labels[idx, ...]),
                  'id': self.ids[idx]}
        sample = self.transform(sample)
        return sample
    
    def extract_train_val_ids(self, n_imgs, seed):
        """
        
        :param n_imgs:
        :param seed:
        :return:
        """
        np.random.seed(seed)  # seed numpy to always get the same images for the same seed
        ids = np.arange(n_imgs)
        np.random.shuffle(ids)  # shuffle inplace
        if self.mode == "train":
            ids = ids[0:int(np.round(len(ids)*self.train_split/100))]
        elif self.mode in ["val", "eval"]:
            ids = ids[int(np.round(len(ids)*self.train_split/100)):]
        else:  # use all ids
            pass
        return ids

    def get_counts(self, counts):
        """

        :param counts:
        :type counts: pandas DataFrame
        :return: sorted nuclear composition DataFrame
        """
        total_counts = counts.iloc[self.ids].sum(axis=0)
        total_counts.name = "counts"
        total_counts.to_csv(self.root_dir / f"total_counts_{self.mode}_{self.train_split}.csv", index=False)
        counts = counts.iloc[self.ids]
        counts = counts.sort_index()
        counts.to_csv(self.root_dir / f"counts_{self.mode}_{self.train_split}.csv", index=False)

        return counts
