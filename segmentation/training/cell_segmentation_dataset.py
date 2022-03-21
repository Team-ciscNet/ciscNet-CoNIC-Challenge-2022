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
        :param train_split: percent of the data used for training and validation (rest ist for test set)
            :type train_split: int
        :return: Dict (image, cell_label, border_label, id).
        """

        imgs = np.load(root_dir/"images.npy")
        if mode in ['train', 'val']:
            labels = np.load(root_dir / "labels.npy")
        else:  # Load GTs for evalation
            labels = np.load(root_dir / "gts.npy").astype(np.int64)
        counts = pd.read_csv(root_dir / "counts.csv")

        self.root_dir = root_dir
        self.mode = mode
        self.train_split = train_split
        self.ids = self.get_train_val_test_split(imgs.shape[0], 0)
        self.imgs = imgs[self.ids, ...]
        self.labels = labels[self.ids, ...]
        self.counts = self.get_counts(counts=counts)
        self.len = len(self.ids)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {'image': np.copy(self.imgs[idx, ...]),
                  'label': np.copy(self.labels[idx, ...]),
                  'id': self.ids[idx]}
        sample = self.transform(sample)
        return sample
    
    def get_train_val_test_split(self, n_imgs, seed):
        """
        
        :param n_imgs:
        :param seed:
        :return:
        """
        np.random.seed(seed)  # seed numpy to always get the same images for the same seed
        ids = np.arange(n_imgs)
        np.random.shuffle(ids)  # shuffle inplace
        if self.mode in ["train", "val"]:
            ids = ids[0:int(np.round(len(ids)*self.train_split/100))]
            if self.mode == 'train':
                ids = ids[0:int(np.round(len(ids) * 80/100))]
            else:
                ids = ids[int(np.round(len(ids) * 80 / 100)):]
        elif self.mode == 'eval':
            ids = sorted(ids[int(np.round(len(ids)*self.train_split/100)):])  # sort for clean eval output
        else:  # use all ids
            pass
        pd.DataFrame({'ids': ids}).to_csv(self.root_dir / f"ids_{self.mode}_{self.train_split}.csv")
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
