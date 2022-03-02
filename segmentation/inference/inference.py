import gc
import json
import pandas as pd
import tifffile as tiff
import torch

from itertools import product

from segmentation.inference.postprocessing import mc_distance_postprocessing, count_nuclei
from segmentation.utils.unets import build_unet
import numpy as np
import ttach as tta


def inference_2d(model, dataset, result_path, device, batchsize, args, use_tta=False, num_gpus=None, mode=None):
    """ Inference function for 2D Cell data sets.

    :param model: Path to the model to use for inference.
        :type model: pathlib Path object.
    :param dataset: Pytorch dataset inference is performed on.
    :param result_path: Path to the results directory.
        :type result_path: pathlib Path object
    :param device: Use (multiple) GPUs or CPU.
        :type device: torch device
    :param batchsize: Batch size.
        :type batchsize: int
    :param args: Arguments for post-processing.
        :type args:
    :param num_gpus: Number of GPUs to use in GPU mode (enables larger batches)
        :type num_gpus: int
    :param mode: Mode used for the dataset ("None" uses all data)
        :type mode: str
    :return: None
    """

    # Load model json file to get architecture + filters
    with open(model.parent / (model.stem + '.json')) as f:
        model_settings = json.load(f)

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

    # Get thresholds / threshold combinations to use
    ths = list(product(args.th_cell if isinstance(args.th_cell, list) else [args.th_cell],
                       args.th_seed if isinstance(args.th_seed, list) else [args.th_seed]))
    for th in ths:
        (result_path / "{}_{}".format(th[0], th[1])).mkdir(exist_ok=True)
    if args.save_raw_pred:
        (result_path / "raws").mkdir(exist_ok=True)

    # Configure dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchsize,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0)

    # Counts dict
    counts = {}
    for th in ths:
        counts[str(th)] = []
    counts['image'] = []

    # Predict images (iterate over images/files)
    for sample in dataloader:

        img_batch, sample_id = sample["image"], sample["id"]
        img_batch = img_batch.to(device)
        sample_id = sample_id.cpu().numpy()

        if mode == 'eval':
            label_batch = sample["label"]
        # Prediction
        if use_tta:
            prediction_batch = net_augm(img_batch)
        else:
            prediction_batch = net(img_batch)

        # to numpy array postprocessing
        prediction_batch = prediction_batch.cpu().numpy()

        # Go through predicted batch and apply post-processing (not parallelized)
        for h in range(len(prediction_batch)):

            print(f'         ... processing {sample_id[h]} ...')

            file_id = str(sample_id[h])

            counts['image'].append(int(file_id))

            for th in ths:

                prediction = mc_distance_postprocessing(np.copy(prediction_batch[h]),
                                                        th_cell=float(th[0]),
                                                        th_seed=float(th[1]),
                                                        downsample=args.upsample)

                tiff.imsave(str(result_path / "{}_{}".format(th[0], th[1]) / f"pred{file_id}.tif"), prediction,
                            compress=1)
                if mode == 'eval':
                    tiff.imsave(str(result_path / "{}_{}".format(th[0], th[1]) / f"gt{file_id}.tif"),
                                label_batch[h, ...].numpy().astype(np.uint16),
                                compress=1)
                if args.save_raw_pred:
                    tiff.imsave(str(result_path / "raws" / f"raw{file_id}.tif"), prediction_batch[h, ...].astype(np.float32),
                                compress=1)

                # Cell counting
                counts[str(th)].append(count_nuclei(prediction))

    counts_df = pd.DataFrame(counts)

    # Sort data frame, remove image id column
    counts_df = counts_df.sort_values('image').drop(columns='image')
    # Save for each th in corresponding folder
    for th in ths:
        df = counts_df[str(th)]
        df = pd.DataFrame(df.to_list(), columns=['neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil',
                                                 'connective'])
        df.to_csv(str(result_path / "{}_{}".format(th[0], th[1]) / "counts.csv"), index=False)

    # Clear memory
    del net
    gc.collect()

    return None
