import numpy as np
import random
import torch
from imgaug import augmenters as iaa
from skimage.filters import gaussian
from torchvision import transforms

from segmentation.utils.utils import min_max_normalization


def augmentors(min_value, max_value):
    """ Get augmentations for the training process.

    :param min_value: Minimum value for the min-max normalization.
        :type min_value: int
    :param max_value: Minimum value for the min-max normalization.
        :type min_value: int
    :return: Dict of augmentations.
    """

    data_transforms = {'train': transforms.Compose([Flip(),
                                                    Color(p=0.55),
                                                    Contrast(p=0.25),
                                                    Scaling(p=0.25),
                                                    Rotate(p=0.25),
                                                    Blur(p=0.33),
                                                    Noise(p=0.33),
                                                    ToTensor(min_value=min_value, max_value=max_value)
                                                    ]),
                       'val': ToTensor(min_value=min_value, max_value=max_value)}

    return data_transforms


class Blur(object):
    """ Blur augmentation (label-preserving transformation) """

    def __init__(self, p=1):
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:
            input_img_dtype = sample['image'].dtype
            input_img_max = np.iinfo(input_img_dtype).max
            sigma = 0.5 * random.random() + 0.4
            sample['image'] = np.round(
                gaussian(sample["image"],  sigma=sigma, channel_axis=-1) * input_img_max
            ).astype(input_img_dtype)
        
        return sample


class Color(object):
    """ Color augmentation (label-preserving transformation) """

    def __init__(self, p=1):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:
            # augmenter that adds a random value sampled uniformly from the range [-20, 20] to the hue and multiplies
            # the saturation by a random factor sampled uniformly from [0.725, 1.275]. It also modifies the contrast of
            # the saturation channel. After these steps, the HSV image is converted back to RGB.
            seq = iaa.Sequential([iaa.WithHueAndSaturation([iaa.WithChannels(0, iaa.Add((-22, 22))),
                                                            iaa.WithChannels(1, [iaa.Multiply((0.7, 1.3)),
                                                                                 iaa.LinearContrast((0.7, 1.2))
                                                                                 ])
                                                            ])
                                  ])
            sample['image'] = seq.augment_image(sample['image'])

        return sample


class Contrast(object):
    """ Contrast augmentation (label-preserving transformation) """

    def __init__(self, p=1):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

            input_img_dtype = sample['image'].dtype
            input_img_min, input_img_max = np.iinfo(input_img_dtype).min, np.iinfo(input_img_dtype).max

            sample['image'] = (sample['image'].astype(np.float32) - input_img_min) / (input_img_max - input_img_min)
            # Really small changes only (otherwise may affect classification)
            contrast_range, gamma_range = (0.9, 1.05), (0.9, 1.05)

            # Contrast
            img_mean, img_min, img_max = sample['image'].mean(), sample['image'].min(), sample['image'].max()
            factor = np.random.uniform(contrast_range[0], contrast_range[1])
            sample['image'] = (sample['image'] - img_mean) * factor + img_mean

            # Gamma
            if random.randint(0, 1) == 0:
                img_mean, img_std, img_min, img_max = sample['image'].mean(), sample['image'].std(), sample['image'].min(), sample['image'].max()
                gamma = np.random.uniform(gamma_range[0], gamma_range[1])
                rnge = img_max - img_min
                sample['image'] = np.power(((sample['image'] - img_min) / float(rnge + 1e-7)), gamma) * rnge + img_min

            sample['image'] = np.clip(sample['image'], 0, 1)
            sample['image'] = sample['image'] * (input_img_max - input_img_min) - input_img_min
            sample['image'] = np.round(sample['image']).astype(input_img_dtype)

        return sample


class Flip(object):
    """ Flip and rotation augmentation (label-preserving transformation). Crop needed for non-square images. """

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        h = random.randint(0, 7)

        if h == 0:  # Original image
            pass

        elif h == 1:  # Flip left-right

            sample['image'] = np.flip(sample['image'], axis=1).copy()
            sample['label'] = np.flip(sample['label'], axis=1).copy()

        elif h == 2:  # Flip up-down

            sample['image'] = np.flip(sample['image'], axis=0).copy()
            sample['label'] = np.flip(sample['label'], axis=0).copy()

        elif h == 3:  # Rotate 90°

            sample['image'] = np.rot90(sample['image'], axes=(0, 1)).copy()
            sample['label'] = np.rot90(sample['label'], axes=(0, 1)).copy()

        elif h == 4:  # Rotate 180°

            sample['image'] = np.rot90(sample['image'], k=2, axes=(0, 1)).copy()
            sample['label'] = np.rot90(sample['label'], k=2, axes=(0, 1)).copy()

        elif h == 5:  # Rotate 270°

            sample['image'] = np.rot90(sample['image'], k=3, axes=(0, 1)).copy()
            sample['label'] = np.rot90(sample['label'], k=3, axes=(0, 1)).copy()

        elif h == 6:  # Flip left-right + rotate 90°

            sample['image'] = np.flip(sample['image'], axis=1).copy()
            sample['image'] = np.rot90(sample['image'], axes=(0, 1)).copy()
            label_img = np.flip(sample['label'], axis=1).copy()
            sample['label'] = np.rot90(label_img, k=1, axes=(0, 1)).copy()

        elif h == 7:  # Flip up-down + rotate 90°

            sample['image'] = np.flip(sample['image'], axis=0).copy()
            sample['image'] = np.rot90(sample['image'], axes=(0, 1)).copy()
            sample['label'] = np.flip(sample['label'], axis=0).copy()
            sample['label'] = np.rot90(sample['label'], k=1, axes=(0, 1)).copy()

        return sample


class Noise(object):
    """ Gaussian noise augmentation """

    def __init__(self, p=0.25):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

            # Add noise with sigma 1-5% of image maximum
            sigma = random.randint(1, 5) / 100 * np.max(sample['image'])
            # Add noise to selected images
            if random.randint(0, 1) == 0:
                seq = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=sigma, per_channel=False)])
            else:
                seq = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=sigma, per_channel=True)])
            sample['image'] = seq.augment_image(sample['image'])

        return sample

 
class Rotate(object):
    """ Rotation augmentation (label-changing augmentation) """

    def __init__(self, p=1):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p
        self.angle = (-45, 45)
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

            angle = random.uniform(self.angle[0], self.angle[1])
                
            seq = iaa.Sequential([iaa.Affine(rotate=angle)]).to_deterministic()
            sample['image'] = seq.augment_image(sample['image'])
            sample['label'] = seq.augment_image(sample['label'])
         
        return sample


class Scaling(object):
    """ Scaling augmentation (label-changing transformation) """

    def __init__(self, p=1):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p
        self.scale = (0.925, 1.075)

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        if random.random() < self.p:

            # Use same scale for both dimensions (otherwise could affect the classification if shape matters)
            scale1 = random.uniform(self.scale[0], self.scale[1])
            # scale2 = random.uniform(self.scale[0], self.scale[1])
            scale2 = scale1

            seq = iaa.Sequential([iaa.Affine(scale={"x": scale1, "y": scale2})])
            sample['image'] = seq.augment_image(sample['image']).copy()
            sample['label'] = seq.augment_image(sample['label']).copy()

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
                if len(sample[key].shape) == 3:  #color channel is present
                    sample[key] = np.transpose(sample[key], (2, 0, 1))
                # color channel not present
                else:
                    sample[key] = sample[key][np.newaxis, ...]

        sample['image'] = torch.from_numpy(sample['image']).to(torch.float)

        # loss function (l1loss/l2loss) needs float tensor with shape [batch, channels, height, width]
        sample['label'] = torch.from_numpy(sample['label']).to(torch.float)

        return sample
