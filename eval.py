import argparse
import numpy as np
import random
import pandas as pd
import tifffile
import torch

from copy import deepcopy
from itertools import product
from pathlib import Path

from segmentation.inference.inference import inference_2d
from segmentation.training.create_training_sets import create_conic_training_sets
from segmentation.utils.metrics import get_conic_metrics, get_perfect_class_metric, get_multi_r2
from segmentation.training.cell_segmentation_dataset import ConicDataset
from segmentation.training.mytransforms import ToTensor


def main():

    random.seed()
    np.random.seed()

    # Get arguments
    parser = argparse.ArgumentParser(description='Conic Challenge - Evaluation')
    parser.add_argument('--model', '-m', required=True, type=str, help='Model to use')
    parser.add_argument('--batch_size', '-bs', default=8, type=int, help='Batch size')
    parser.add_argument('--multi_gpu', '-mgpu', default=False, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--save_raw_pred', '-srp', default=False, action='store_true', help='Save raw predictions')
    parser.add_argument('--th_cell', '-tc', default=0.07, nargs='+', help='Threshold for adjusting cell size')
    parser.add_argument('--th_seed', '-ts', default=0.45, nargs='+', help='Threshold for seeds')
    parser.add_argument('--tta', '-tta', default=False, action='store_true', help='Use test-time augmentation')
    parser.add_argument('--upsample', '-u', default=False, action='store_true', help='Apply rescaling (1.25) for inference')
    parser.add_argument('--calc_perfect_class_metric', '-cpcm', default=False, action='store_true',
                        help='Calculate metric for predicted segmentation and ground truth classification')
    args = parser.parse_args()

    # Paths
    path_data = Path(__file__).parent / 'training_data' / 'conic_fixed_train_valid'
    path_models = Path(__file__).parent / 'models'
    if args.upsample:
        path_train_data = path_data / 'upsampled'
    else:
        path_train_data = path_data / 'original_scale'

    # Set device for using CPU or GPU
    device, num_gpus = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 1
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.multi_gpu:
        num_gpus = torch.cuda.device_count()
        
    # Check if training data (labels_train.npy) already exist
    if not (path_train_data / 'train_labels.npy').is_file() or not (path_train_data / 'valid_labels.npy').is_file():
        # Create training sets
        print(f'No training data found. Creating training data.\nUse upsampling: {args.upsample}')
        if not (path_data / 'train_imgs.npy').is_file():
            raise Exception('train_imgs.npy not found in {}'.format(path_data))
        if not (path_data / 'train_anns.npy').is_file():
            raise Exception('train_anns.npy not found in {}'.format(path_data))
        if not (path_data / 'valid_imgs.npy').is_file():
            raise Exception('valid_imgs.npy not found in {}'.format(path_data))
        if not (path_data / 'valid_anns.npy').is_file():
            raise Exception('valid_anns.npy not found in {}'.format(path_data))
        path_train_data.mkdir(exist_ok=True)
        create_conic_training_sets(path_data=path_data, path_train_data=path_train_data, upsample=args.upsample,
                                   mode='train')
        create_conic_training_sets(path_data=path_data, path_train_data=path_train_data, upsample=args.upsample,
                                   mode='valid')

    # Load model
    model = path_models / "{}.pth".format(args.model)

    # Directory for results
    path_seg_results = path_train_data / f"{model.stem}"
    path_seg_results.mkdir(exist_ok=True)
    print(f"Evaluation of {model.stem}. Seed thresholds: {args.th_seed}, mask thresholds: {args.th_cell}, "
          f"upsampling: {args.upsample}, tta: {args.tta}")

    inference_args = deepcopy(args)
    
    dataset = ConicDataset(root_dir=path_train_data,
                           mode="eval",
                           transform=ToTensor(min_value=0, max_value=255))

    inference_2d(model=model,
                 dataset=dataset,
                 result_path=path_seg_results,
                 device=device,
                 batchsize=args.batch_size,
                 args=inference_args,
                 num_gpus=num_gpus,
                 use_tta=args.tta,
                 mode='eval')  # uses validation set
    
    # Calculate metrics
    ths = list(product(args.th_cell if isinstance(args.th_cell, list) else [args.th_cell],
                       args.th_seed if isinstance(args.th_seed, list) else [args.th_seed]))
    for th in ths:
        path_seg_results_th = path_seg_results / "{}_{}".format(th[0], th[1])

        pred_ids = list(path_seg_results_th.glob('pred*.tif'))

        preds, gts = [], []
        for pred_id in pred_ids:
            preds.append(tifffile.imread(str(pred_id)))
            gts.append(tifffile.imread(str(pred_id.parent / "gt{}".format(pred_id.name.split('pred')[-1]))))

        preds = np.transpose(np.array(preds), (0, 2, 3, 1))
        gts = np.transpose(np.array(gts), (0, 2, 3, 1))

        print(f"Calculate metrics (upsampling: {args.upsample}, th_cell: {th[0]}, th_seed: {th[1]}):")
        metrics = np.squeeze(get_conic_metrics(gts, preds).values)
        if args.calc_perfect_class_metric:
            print(f"Calculate metrics for prediction with ground truth classification:")
            metrics_perfect_class = get_perfect_class_metric(gts, preds)[0]
        else:
            metrics_perfect_class = -1

        result = pd.DataFrame([[args.model, args.upsample, th[0], th[1], metrics[0], metrics[1],
                                metrics_perfect_class, args.tta]],
                              columns=["model_name", "upsampling", "th_cell", "th_seed", "multi_pq+", "pq_metrics_avg",
                                       "multi_pq+_perfect_class", "tta"])
        
        result.to_csv(Path(__file__).parent / "scores_post-challenge-analysis.csv",
                      header=not (Path(__file__).parent / "scores_post-challenge-analysis.csv").exists(),
                      index=False,
                      mode="a")


if __name__ == "__main__":
    main()
