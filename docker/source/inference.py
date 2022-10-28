import gc
import json
import numpy as np
import pandas as pd
import torch
import ttach as tta


from source.postprocessing import mc_distance_postprocessing, count_nuclei
from source.unets import build_unet


def inference_2d(res_size, model, dataset, device, batchsize, th_cell, th_seed, downsample, use_tta=False,
                 num_gpus=None):
    """ Inference function for 2D Cell data sets.

    :param model: Path to the model to use for inference.
        :type model: pathlib Path object.
    :param dataset: Pytorch dataset inference is performed on.
    :param device: Use (multiple) GPUs or CPU.
        :type device: torch device
    :param batchsize: Batch size.
        :type batchsize: int
    :param num_gpus: Number of GPUs to use in GPU mode (enables larger batches)
        :type num_gpus: int
    :return: None
    """

    # Load model json file to get architecture + filters
    with open(model.parent / (model.stem + '.json')) as f:
        model_settings = json.load(f)

    # Allocate results array
    results = np.zeros(shape=res_size, dtype=np.uint16)

    # Build model
    net = build_unet(act_fun=model_settings['architecture'][2],
                     pool_method=model_settings['architecture'][1],
                     normalization=model_settings['architecture'][3],
                     device=device,
                     num_gpus=num_gpus,
                     ch_in=model_settings['architecture'][5],
                     ch_out=model_settings['architecture'][6],
                     filters=model_settings['architecture'][4])

    # Get number of GPUs to use and load weights
    if not num_gpus:
        num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        net.module.load_state_dict(torch.load(str(model), map_location=device))
    else:
        net.load_state_dict(torch.load(str(model), map_location=device))
    net.eval()
    if use_tta:
        net_augm = tta.SegmentationTTAWrapper(net, tta.aliases.d4_transform(), merge_mode="mean")
    torch.set_grad_enabled(False)

    # Configure dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchsize,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0)

    # Counts dict
    counts = []

    # Predict images (iterate over images/files)
    for sample in dataloader:

        img_batch, sample_id = sample["image"], sample["id"]
        img_batch = img_batch.to(device)
        sample_id = sample_id.cpu().numpy()

        # Prediction
        if use_tta:
            prediction_batch = net_augm(img_batch)
        else:
            prediction_batch = net(img_batch)

        # to numpy array postprocessing
        prediction_batch = prediction_batch.cpu().numpy()

        # Go through predicted batch and apply post-processing (not parallelized)
        for h in range(len(prediction_batch)):

            results[sample_id[h]] = mc_distance_postprocessing(np.copy(prediction_batch[h]),
                                                               th_cell=float(th_cell),
                                                               th_seed=float(th_seed),
                                                               downsample=downsample)

            # Cell counting
            counts.append(count_nuclei(np.copy(results[sample_id[h]])))

    counts_df = pd.DataFrame(counts, columns=['neutrophil', 'epithelial-cell', 'lymphocyte', 'plasma-cell',
                                              'eosinophil', 'connective-tissue-cell'])

    # Clear memory
    del net
    gc.collect()

    return results, counts_df
