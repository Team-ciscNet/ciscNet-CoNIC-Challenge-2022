import numpy as np
import pandas as pd

from copy import deepcopy
from skimage import measure
from scipy.optimize import linear_sum_assignment
from skimage.measure import label
from sklearn.metrics import r2_score


def get_fast_aji_plus(true, pred):
    """Adapted from https://github.com/vqdang/hover_net/blob/master/metrics/stats_utils.py under MIT Licence
    
    MIT License

    Copyright (c) 2020 vqdang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    AJI+, an AJI version with maximal unique pairing to obtain overall intersecion.
    Every prediction instance is paired with at most 1 GT instance (1 to 1) mapping, unlike AJI 
    where a prediction instance can be paired against many GT instances (1 to many).
    Remaining unpaired GT and Prediction instances will be added to the overall union.
    The 1 to 1 mapping prevents AJI's over-penalisation from happening.
    Empty true images result in return NAN regardless of pred!
    """

    true = np.copy(true)  # ? do we need this
    true = measure.label(true)
    pred = np.copy(pred)
    pred = measure.label(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    #### Munkres pairing to find maximal unique pairing
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    ### extract the paired cost and remove invalid pair
    paired_iou = pairwise_iou[paired_true, paired_pred]
    # now select all those paired with iou != 0.0 i.e have intersection
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]
    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score


def get_multi_pq_info(true, pred, nr_classes=6, match_iou=0.5):
    """ From https://github.com/TissueImageAnalytics/CoNIC/blob/main/metrics/stats_utils.py
    
    Get the statistical information needed to compute multi-class PQ.
    
    CoNIC multiclass PQ is achieved by considering nuclei over all images at the same time, 
    rather than averaging image-level results, like was done in MoNuSAC. This overcomes issues
    when a nuclear category is not present in a particular image.
    
    Args:
        true (ndarray): HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map. 
        pred: HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map. 
        nr_classes (int): Number of classes considered in the dataset. 
        match_iou (float): IoU threshold for determining whether there is a detection.
    
    Returns:
        statistical info per class needed to compute PQ.
    
    """

    assert match_iou >= 0.0, "Cant' be negative"

    true_inst = true[..., 0]
    pred_inst = pred[..., 0]
    ###
    true_class = true[..., 1]
    pred_class = pred[..., 1]

    pq = []
    for idx in range(nr_classes):
        pred_class_tmp = pred_class == idx + 1
        pred_inst_oneclass = pred_inst * pred_class_tmp
        pred_inst_oneclass = remap_label(pred_inst_oneclass)
        ##
        true_class_tmp = true_class == idx + 1
        true_inst_oneclass = true_inst * true_class_tmp
        true_inst_oneclass = remap_label(true_inst_oneclass)

        pq_oneclass_info = get_pq(true_inst_oneclass, pred_inst_oneclass, remap=False)

        # add (in this order) tp, fp, fn iou_sum
        pq_oneclass_stats = [
            pq_oneclass_info[1][0],
            pq_oneclass_info[1][1],
            pq_oneclass_info[1][2],
            pq_oneclass_info[2],
        ]
        pq.append(pq_oneclass_stats)

    return pq


def get_pq(true, pred, match_iou=0.5, remap=True):
    """ From https://github.com/TissueImageAnalytics/CoNIC/blob/main/metrics/stats_utils.py
    
    Get the panoptic quality result. 
    
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` beforehand. Here, the `by_size` flag 
    has no effect on the result.
    Args:
        true (ndarray): HxW ground truth instance segmentation map
        pred (ndarray): HxW predicted instance segmentation map
        match_iou (float): IoU threshold level to determine the pairing between
            GT instances `p` and prediction instances `g`. `p` and `g` is a pair
            if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
            (1 prediction instance to 1 GT instance mapping). If `match_iou` < 0.5, 
            Munkres assignment (solving minimum weight matching in bipartite graphs) 
            is caculated to find the maximal amount of unique pairing. If 
            `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
            the number of pairs is also maximal.  
        remap (bool): whether to ensure contiguous ordering of instances.
    
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
        
        paired_iou.sum(): sum of IoU within true positive predictions
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"
    # ensure instance maps are contiguous
    if remap:
        pred = remap_label(pred)
        true = remap_label(true)

    true = np.copy(true)
    pred = np.copy(pred)
    true = true.astype("int32")
    pred = pred.astype("int32")
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask_lab = true == true_id
        rmin1, rmax1, cmin1, cmax1 = get_bounding_box(t_mask_lab)
        t_mask_crop = t_mask_lab[rmin1:rmax1, cmin1:cmax1]
        t_mask_crop = t_mask_crop.astype("int")
        p_mask_crop = pred[rmin1:rmax1, cmin1:cmax1]
        pred_true_overlap = p_mask_crop[t_mask_crop > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask_lab = pred == pred_id
            p_mask_lab = p_mask_lab.astype("int")

            # crop region to speed up computation
            rmin2, rmax2, cmin2, cmax2 = get_bounding_box(p_mask_lab)
            rmin = min(rmin1, rmin2)
            rmax = max(rmax1, rmax2)
            cmin = min(cmin1, cmin2)
            cmax = max(cmax1, cmax2)
            t_mask_crop2 = t_mask_lab[rmin:rmax, cmin:cmax]
            p_mask_crop2 = p_mask_lab[rmin:rmax, cmin:cmax]

            total = (t_mask_crop2 + p_mask_crop2).sum()
            inter = (t_mask_crop2 * p_mask_crop2).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / ((tp + 0.5 * fp + 0.5 * fn) + 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return (
        [dq, sq, dq * sq],
        [tp, fp, fn],
        paired_iou.sum(),
    )


def remap_label(pred, by_size=False):
    """ From https://github.com/TissueImageAnalytics/CoNIC/blob/main/misc/utils.py
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.
    Args:
        pred (ndarray): the 2d array contain instances where each instances is marked
            by non-zero integer.
        by_size (bool): renaming such that larger nuclei have a smaller id (on-top).
    Returns:
        new_pred (ndarray): Array with continguous ordering of instances.
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def get_bounding_box(img):
    """ From https://github.com/TissueImageAnalytics/CoNIC/blob/main/misc/utils.py
    Get the bounding box coordinates of a binary input- assumes a single object.
    Args:
        img: input binary image.
    Returns:
        bounding box coordinates
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def get_conic_metrics(trues, preds):
    """Adapted from https://github.com/TissueImageAnalytics/CoNIC/blob/main/compute_stats.py

    Args:
        trues (numpy array): Numpy array with NxHxWx2, N: Number of images, 2: first dimension instance, second dimension class
        preds (numpy array): Numpy array with NxHxWx2, N: Number of images, 2: first dimension instance, second dimension class
    """

    all_metrics = {}
    # initialise empty placeholder lists
    pq_list = []
    mpq_info_list = []
    # load the prediction and ground truth arrays

    nr_patches = preds.shape[0]

    for patch_idx in range(nr_patches):
        # get a single patch
        pred = preds[patch_idx]
        true = trues[patch_idx]

        # instance segmentation map
        pred_inst = pred[..., 0]
        true_inst = true[..., 0]
        # classification map
        pred_class = pred[..., 1]
        true_class = true[..., 1]

        # ===============================================================
        # get binary panoptic quality
        pq = get_pq(true_inst, pred_inst)
        pq = pq[0][2]
        pq_list.append(pq)

        # get the multiclass pq stats info from single image
        mpq_info_single = get_multi_pq_info(true, pred)
        mpq_info = []
        # aggregate the stat info per class
        for single_class_pq in mpq_info_single:
            tp = single_class_pq[0]
            fp = single_class_pq[1]
            fn = single_class_pq[2]
            sum_iou = single_class_pq[3]
            mpq_info.append([tp, fp, fn, sum_iou])
        mpq_info_list.append(mpq_info)

    pq_metrics = np.array(pq_list)
    pq_metrics_avg = np.mean(pq_metrics, axis=-1)  # average over all images

    mpq_info_metrics = np.array(mpq_info_list, dtype="float")
    # sum over all the images
    total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)

    mpq_list = []
    # for each class, get the multiclass PQ
    for cat_idx in range(total_mpq_info_metrics.shape[0]):
        total_tp = total_mpq_info_metrics[cat_idx][0]
        total_fp = total_mpq_info_metrics[cat_idx][1]
        total_fn = total_mpq_info_metrics[cat_idx][2]
        total_sum_iou = total_mpq_info_metrics[cat_idx][3]

        # get the F1-score i.e DQ
        dq = total_tp / (
            (total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6
        )
        # get the SQ, when not paired, it has 0 IoU so does not impact
        sq = total_sum_iou / (total_tp + 1.0e-6)
        mpq_list.append(dq * sq)
    mpq_metrics = np.array(mpq_list)
    all_metrics["multi_pq+"] = [np.mean(mpq_metrics)]

    all_metrics["pq_metrics_avg"] = [pq_metrics_avg]

    df = pd.DataFrame(all_metrics)
    print(df.to_string(index=False))
    return df


def get_perfect_class_metric(gts, preds):
    """Return the predictions with perfect classification. Predicted cells without matching ground truth (false positives) will not be deleted but keep
    the original prediction

    Args:
        gts (np.array): Ground truth segmentation and classification (segmentation: [...,0], classification: [...,1])
        preds (np.array): Predicted segmentation and classification  (segmentation: [...,0], classification: [...,1])

    Returns:
        metrics_pc (np.array): Metrics with predicted segmentation and ground truth classification
    """
    gt_class = np.zeros_like(preds[..., 1])
    for idx in range(preds.shape[0]):
        pred_instance = label(preds[idx, ..., 0])
        for i in range(1, pred_instance.max() + 1):
            # get the counts of cell i
            cell_mask = preds[idx, ..., 0] == i
            counts = np.bincount(gts[idx, ..., 1][cell_mask], minlength=7)[1:]
            if counts.max() == 0:
                # Cell not present in groundtruth => false positive => leave class as is
                gt_class[idx, ...][preds[idx, ..., 0] == i] = preds[idx, ..., 1][cell_mask]
            else:
                # Get class from maximum in bincount
                gt_class[idx, ...][preds[idx, ..., 0] == i] = np.argmax(counts) + 1  # +1 since we cut the zero
    preds_pc = deepcopy(preds)
    preds_pc[..., 1] = gt_class
    metrics_pc = np.squeeze(get_conic_metrics(gts, preds_pc).values)

    return metrics_pc


def get_multi_r2(true, pred):
    """
    https://github.com/TissueImageAnalytics/CoNIC/blob/main/metrics/stats_utils.py

    Get the correlation of determination for each class and then
    average the results.

    Args:
        true (pd.DataFrame): dataframe indicating the nuclei counts for each image and category.
        pred (pd.DataFrame): dataframe indicating the nuclei counts for each image and category.

    Returns:
        multi class coefficient of determination

    """
    # first check to make sure that the appropriate column headers are there
    class_names = [
        "epithelial",
        "lymphocyte",
        "plasma",
        "neutrophil",
        "eosinophil",
        "connective",
    ]
    for col in true.columns:
        if col not in class_names:
            raise ValueError("%s column header not recognised")

    for col in pred.columns:
        if col not in class_names:
            raise ValueError("%s column header not recognised")

    # for each class, calculate r2 and then take the average
    r2_list = []
    for class_ in class_names:
        true_oneclass = true[class_].tolist()
        pred_oneclass = pred[class_].tolist()
        r2_list.append(r2_score(true_oneclass, pred_oneclass))

    return np.mean(np.array(r2_list))
