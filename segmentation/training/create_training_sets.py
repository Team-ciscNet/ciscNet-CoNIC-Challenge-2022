import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
from skimage.transform import rescale

from segmentation.training.train_data_representations import distance_label


def create_conic_training_sets(path_data, path_train_data, upsample, mode):
    """ Create training sets for CoNIC Challenge data.

    :param path_data: Path to the directory containing the CoNIC Challenge data / training data.
        :type path_data: Pathlib Path object.
    :param path_train_data: Path to the directory to save the created training data into.
        :type path_data: Pathlib Path object.
    :param upsample: Apply upsampling (factor 1.25).
        :type upsample: bool
    :param mode: 'train' or 'valid'
        :type mode: str
    :return: None
    """

    print(f"Create data for mode {mode}.")

    imgs = np.load(path_data / f"{mode}_imgs.npy")
    gts = np.load(path_data / f"{mode}_anns.npy")

    if upsample:  # results for conic patches in 320-by-320 patches
        scale = 1.25
        imgs_scaled = np.zeros(shape=(imgs.shape[0],
                                      int(scale * imgs.shape[1]),
                                      int(scale * imgs.shape[2]),
                                      imgs.shape[3]), dtype=imgs.dtype)
        gts_scaled = np.zeros(shape=(gts.shape[0],
                                     int(scale * gts.shape[1]),
                                     int(scale * gts.shape[2]),
                                     gts.shape[3]), dtype=gts.dtype)
        print(f"Scale images:")
        for i in tqdm(range(len(imgs))):
            imgs_scaled[i] = rescale(imgs[i],
                                     scale=(scale, scale, 1),
                                     order=2,
                                     anti_aliasing=True,
                                     preserve_range=True).astype(imgs.dtype)
            gts_scaled[i] = rescale(gts[i],
                                    scale=(scale, scale, 1),
                                    order=0,
                                    anti_aliasing=False,
                                    preserve_range=True).astype(gts.dtype)
        imgs = imgs_scaled
        labels = gts_scaled
    else:
        labels = gts

    labels_train = np.empty((*imgs.shape[0:3], 7), dtype=np.float32)  # 6 classes + combination
    print(f"Create distance labels:")
    for label_id in tqdm(range(labels.shape[0])):
        labels_train[label_id, ..., 0] = distance_label(labels[label_id, ..., 0])

    for i in range(1, 7):
        labels_train[..., i] = labels_train[..., 0] * (labels[..., 1] == i)

    # Detect slices with no cells
    slice_ids = []
    print(f"Remove slices without cells:")
    for i in tqdm(range(labels.shape[0])):
        if np.max(labels_train[i, ..., 0]) < 1:
            slice_ids.append(i)

    imgs = np.delete(imgs, np.array(slice_ids), axis=0)
    labels_train = np.delete(labels_train, np.array(slice_ids), axis=0)
    gts = np.delete(gts, np.array(slice_ids), axis=0)

    np.save(path_train_data / f"{mode}_images.npy", imgs)
    np.save(path_train_data / f"{mode}_labels.npy", labels_train)
    np.save(path_train_data / f"{mode}_gts.npy", gts)

    # save tiffs for imagej visualization
    tifffile.imsave(path_train_data / f"{mode}_labels_channel_0.tiff", labels_train[..., 0])
    tifffile.imsave(path_train_data / f"{mode}_labels_channel_1.tiff", labels_train[..., 1])
    tifffile.imsave(path_train_data / f"{mode}_labels_channel_2.tiff", labels_train[..., 2])
    tifffile.imsave(path_train_data / f"{mode}_labels_channel_3.tiff", labels_train[..., 3])
    tifffile.imsave(path_train_data / f"{mode}_labels_channel_4.tiff", labels_train[..., 4])
    tifffile.imsave(path_train_data / f"{mode}_labels_channel_5.tiff", labels_train[..., 5])
    tifffile.imsave(path_train_data / f"{mode}_labels_channel_6.tiff", labels_train[..., 6])
    tifffile.imsave(path_train_data / f"{mode}_gts_instance.tiff", gts[..., 0])
    tifffile.imsave(path_train_data / f"{mode}_gts_class.tiff", gts[..., 1])
    tifffile.imsave(path_train_data / f"{mode}_images.tiff", imgs)
    
    return None
