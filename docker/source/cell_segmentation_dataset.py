import numpy as np
import torch

from skimage.transform import rescale
from torch.utils.data import Dataset
from torchvision import transforms

from source.utils import min_max_normalization


class ConicDataset(Dataset):
    """ Pytorch data set for CoNIC Challenge """

    def __init__(self, root_dir, transform=lambda x: x):
        """

        :param root_dir: Directory containing the dataset.
            :type root_dir: pathlib Path object.
        :param transform: transforms.
            :type transform:
        :return: Dict (image, cell_label, border_label, id).
        """

        imgs = np.load(root_dir/"images.npy")

        self.root_dir = root_dir
        self.imgs = imgs
        self.ids = np.arange(len(imgs))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = {'image': np.copy(self.imgs[idx, ...]),
                  'id': self.ids[idx]}
        sample = self.transform(sample)
        return sample


class ToTensor(object):
    """ Convert image and label image to Torch tensors """

    def __init__(self, min_value, max_value):
        """

        :param min_value: Minimum value for the normalization. All values below this value are clipped
            :type min_value: int
        :param max_value: Maximum value for the normalization. All values above this value are clipped.
            :type max_value: int
        """
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        # Normalize image
        sample['image'] = min_max_normalization(sample['image'], min_value=self.min_value, max_value=self.max_value)

        # Swap axes from (H, W, Channels) or (H, W) to (Channels, H, W)
        for key in sample:
            if key != 'id':
                if len(sample[key].shape) == 3:  # color channel is present
                    sample[key] = np.transpose(sample[key], (2, 0, 1))
                # color channel not present
                else:
                    sample[key] = sample[key][np.newaxis, ...]

        sample['image'] = torch.from_numpy(sample['image']).to(torch.float)

        return sample


class Upsample(object):

    def __init__(self, scale=1.25):

        self.scale = scale

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        sample['image'] = rescale(sample['image'],
                                  scale=(self.scale, self.scale, 1),
                                  order=2,
                                  anti_aliasing=True,
                                  preserve_range=True).astype(sample['image'].dtype)

        return sample


def infer_transforms(upsample, min_value, max_value):
    """ Get augmentations for the training process.

    :param min_value: Minimum value for the min-max normalization.
        :type min_value: int
    :param max_value: Minimum value for the min-max normalization.
        :type min_value: int
    :return: Dict of augmentations.
    """

    if upsample:
        data_transforms = transforms.Compose([Upsample(),
                                              ToTensor(min_value=min_value, max_value=max_value)
                                              ])
    else:
        data_transforms = ToTensor(min_value=min_value, max_value=max_value)

    return data_transforms
