import numpy as np
from skimage.segmentation import watershed
from skimage import measure
from skimage.transform import rescale

from source.utils import get_nucleus_ids


def mc_distance_postprocessing(cell_prediction, th_cell, th_seed, downsample):
    """ Post-processing for distance label (cell + neighbor) prediction.

    :param cell_prediction: Multi channel cell distance prediction.
    :type cell_prediction:
    :param th_cell:
    :type th_cell: float
    :param th_seed:
    :type th_seed: float

    :return: Instance segmentation mask.
    """

    min_area = 2  # keep only seeds larger than threshold area

    # Instance segmentation (use summed up channel 0)
    mask = cell_prediction[0] > th_cell  # get binary mask by thresholding distance prediction
    seeds = measure.label(cell_prediction[0] > th_seed, background=0)  # get seeds
    props = measure.regionprops(seeds)  # Remove very small seeds
    for idx, prop in enumerate(props):
        if prop.area < min_area:
            seeds[seeds == prop.label] = 0
    seeds = measure.label(seeds, background=0)
    prediction_instance = watershed(image=-cell_prediction[0], markers=seeds, mask=mask, watershed_line=False)
    
    # Semantic segmentation / classification
    prediction_class = np.zeros_like(prediction_instance)
    for idx in range(1, prediction_instance.max()+1):
        # Get sum of distance prediction of selected cell for each class (class 0 is sum of the other classes)
        pix_vals = cell_prediction[1:][:, prediction_instance == idx]
        cell_layer = np.sum(pix_vals, axis=1).argmax() + 1  # +1 since class 0 needs to be considered for argmax
        prediction_class[prediction_instance == idx] = cell_layer

    if downsample:
        # Downsample instance segmentation
        prediction_instance = rescale(prediction_instance,
                                      scale=0.8,
                                      order=0,
                                      preserve_range=True,
                                      anti_aliasing=False).astype(np.uint16)

        # Downsample semantic segmentation
        prediction_class = rescale(prediction_class,
                                   scale=0.8,
                                   order=0,
                                   preserve_range=True,
                                   anti_aliasing=False).astype(np.uint16)

    # Combine instance segmentation and semantic segmentation results
    prediction = np.concatenate((prediction_instance[np.newaxis, ...], prediction_class[np.newaxis, ...]), axis=0)
    prediction = np.transpose(prediction, (1, 2, 0))

    return prediction.astype(np.uint16)


def count_nuclei(prediction, border_width=16, classes=6):
    """ Count nuclei of each class"""

    border_width += 1

    # Border correction for counting
    prediction = prediction[border_width:prediction.shape[0] - border_width,  border_width:prediction.shape[1] - border_width, :]

    nucleus_ids = get_nucleus_ids(prediction[:, :, 0])

    if len(nucleus_ids) == 0:
        return np.zeros(classes, dtype=int)
    else:
        class_ids = [get_nucleus_ids(prediction[:, :, 1][prediction[:, :, 0] == nucleus_idx]) for nucleus_idx in nucleus_ids]
        h_counts = np.unique(class_ids, return_counts=True)
        # Consider that some classes may not occur in the current image and do not appear in classes
        counts = np.zeros(classes, dtype=int)
        counts[h_counts[0]-1] = h_counts[1]
        return counts

