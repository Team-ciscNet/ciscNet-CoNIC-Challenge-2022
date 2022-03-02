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
    parser.add_argument('--dataset', '-ds', default='conic_patches', type=str, help='"conic_patches" or "lizard"')
    parser.add_argument('--batch_size', '-bs', default=8, type=int, help='Batch size')
    parser.add_argument('--multi_gpu', '-mgpu', default=False, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--save_raw_pred', '-srp', default=False, action='store_true', help='Save raw predictions')
    parser.add_argument('--th_cell', '-tc', default=0.07, nargs='+', help='Threshold for adjusting cell size')
    parser.add_argument('--th_seed', '-ts', default=0.45, nargs='+', help='Threshold for seeds')
    parser.add_argument('--tta', '-tta', default=False, action='store_true', help='Use test-time augmentation')
    parser.add_argument('--eval_split', '-es', default=80, type=int, help='Train split in %')
    parser.add_argument('--upsample', '-u', default=False, action='store_true', help='Apply rescaling (1.25) for inference')
    parser.add_argument('--calc_perfect_class_metric', '-cpcm', default=False, action='store_true',
                        help='Calculate metric for predicted segmentation and ground truth classification')
    args = parser.parse_args()

    # Paths
    path_models = Path(__file__).parent / 'models'
    if args.upsample:
        path_train_data = Path(__file__).parent / 'training_data' / args.dataset / 'upsampled'
    else:
        path_train_data = Path(__file__).parent / 'training_data' / args.dataset / 'original_scale'

    if args.dataset == 'lizard':
        raise NotImplementedError

    # Set device for using CPU or GPU
    device, num_gpus = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 1
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.multi_gpu:
        num_gpus = torch.cuda.device_count()
        
    # Check if data to evaluate exists
    if not (path_train_data / 'images.npy').is_file() or not (path_train_data / 'labels.npy').is_file() \
       or not (path_train_data / 'gts.npy').is_file():
        # Create training sets
        print(f'No training data found. Creating training data.\nUse upsampling: {args.upsample}')
        if not (path_train_data.parent / 'images.npy').is_file():
            raise Exception('images.npy not found in {}'.format(path_train_data.parent))
        if not (path_train_data.parent / 'labels.npy').is_file():
            raise Exception('labels.npy not found in {}'.format(path_train_data.parent))
        path_train_data.mkdir(exist_ok=True)
        create_conic_training_sets(path_data=path_train_data.parent,
                                   path_train_data=path_train_data,
                                   upsample=args.upsample)

    # Load model
    model = path_models / "{}.pth".format(args.model)

    # Directory for results
    path_seg_results = path_train_data / f"{model.stem}_{args.eval_split}"
    path_seg_results.mkdir(exist_ok=True)
    print(f"Evaluation of {model.stem}. Seed thresholds: {args.th_seed}, mask thresholds: {args.th_cell}, "
          f"upsampling: {args.upsample}, tta: {args.tta}")

    inference_args = deepcopy(args)
    
    if args.dataset == "conic_patches":
        dataset = ConicDataset(root_dir=path_train_data,
                               mode="eval",
                               transform=ToTensor(min_value=0, max_value=255),
                               train_split=args.eval_split)
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')

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

        # r2 metric
        pred_counts = pd.read_csv(path_seg_results_th / "counts.csv")
        gt_counts = dataset.counts
        gt_counts = gt_counts.sort_index()
        r2 = get_multi_r2(gt_counts, pred_counts)
        print(f"  R2: {r2}")

        result = pd.DataFrame([[args.model, args.dataset, args.upsample, th[0], th[1], metrics[0], metrics[1],
                                metrics_perfect_class, r2, args.tta]],
                              columns=["model_name", "dataset", "upsampling", "th_cell", "th_seed", "multi_pq+", "pq_metrics_avg",
                                       "multi_pq+_perfect_class", "R2", "tta"])
        
        result.to_csv(Path(__file__).parent / f"scores{args.eval_split}.csv",
                      header=not (Path(__file__).parent / f"scores{args.eval_split}.csv").exists(),
                      index=False,
                      mode="a")


if __name__ == "__main__":
    main()
