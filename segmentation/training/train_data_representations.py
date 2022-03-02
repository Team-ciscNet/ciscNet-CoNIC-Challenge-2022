import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage import measure


def distance_label(label):
    """ Cell and neigbhor distance label creation (Euclidean distance).

    :param label: Intensity-coded instance segmentation label image.
        :type label:

    :return: Cell distance label image, neighbor distance label image.
    """

    # Preallocation
    label_dist = np.zeros(shape=label.shape, dtype=np.float)

    # Find centroids, crop image, calculate distance transforms
    props = measure.regionprops(label)
    for i in range(len(props)):

        # Get nucleus and Euclidean distance transform for each nucleus
        nucleus = (label == props[i].label)
        centroid, mal = np.round(props[i].centroid), int(1.2 * np.ceil(props[i].major_axis_length))
        if mal <= 1:
            continue
        nucleus_crop = nucleus[
                       int(max(centroid[0] - mal, 0)):int(min(centroid[0] + mal, label.shape[0])),
                       int(max(centroid[1] - mal, 0)):int(min(centroid[1] + mal, label.shape[1]))
                       ]
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)
        if np.max(nucleus_crop_dist) > 0:
            nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)
        label_dist[
        int(max(centroid[0] - mal, 0)):int(min(centroid[0] + mal, label.shape[0])),
        int(max(centroid[1] - mal, 0)):int(min(centroid[1] + mal, label.shape[1]))
        ] += nucleus_crop_dist

    return label_dist.astype(np.float32)
