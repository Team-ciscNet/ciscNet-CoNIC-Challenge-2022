import argparse
import numpy as np
import random
import torch

from pathlib import Path

from segmentation.training.cell_segmentation_dataset import ConicDataset
from segmentation.training.create_training_sets import create_conic_training_sets
from segmentation.training.mytransforms import augmentors
from segmentation.training.training import train, get_max_epochs, get_weights

from segmentation.utils import utils
from segmentation.utils.unets import build_unet


def main():
    """ Train models on conic patches or lizard data set. """

    random.seed()
    np.random.seed()

    # Get arguments
    parser = argparse.ArgumentParser(description='Conic Challenge - Training')
    parser.add_argument('--model_name', '-m', default='conic_model', type=str,
                        help='Building block for the unique model name. Best use a suffix, e.g., "conic_model_mb')
    parser.add_argument('--act_fun', '-af', default='relu', type=str, help='Activation function')
    parser.add_argument('--batch_size', '-bs', default=8, type=int, help='Batch size')
    parser.add_argument('--classes', '-c', default=6, type=int, help='Classes to predict')
    parser.add_argument('--filters', '-f', nargs=2, type=int, default=[64, 1024], help='Filters for U-net')
    parser.add_argument('--loss', '-l', default='smooth_l1', type=str, help='Loss function')
    parser.add_argument('--multi_gpu', '-mgpu', default=False, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--norm_method', '-nm', default='bn', type=str, help='Normalization method')
    parser.add_argument('--optimizer', '-o', default='adam', type=str, help='Optimizer')
    parser.add_argument('--pool_method', '-pm', default='conv', type=str, help='Pool method')
    parser.add_argument('--upsample', '-u', default=False, action='store_true', help='Apply rescaling (1.25)')
    parser.add_argument('--channels_in', '-cin', default=3, type=int, help="Number of input channels")
    parser.add_argument('--max_epochs', '-me', default=None, type=int, help='Maximum number of epochs (None: auto defined)')
    parser.add_argument('--loss_fraction_weights', '-lfw', default=None, nargs="+", type=float,
                        help="Weights for weighting the classes (first weight: summed up (main) channel.")
    parser.add_argument('--weightmap_weights', '-ww', default=None, nargs="+", type=float,
                        help="Weights used in the weightmaps for each class (first weight: summed up (main) channel.")
    args = parser.parse_args()
    
    if args.loss_fraction_weights is None:
        args.loss_fraction_weights = list(np.ones(args.classes+1))
    if len(args.loss_fraction_weights) != (args.classes+1):
        parser.error(f"--classes ({args.classes}+1) needs to match --loss_fraction_weights number of arguments "
                     f"({len(args.loss_fraction_weights)})")

    if args.loss == "weighted_smooth_l1":
        if args.weightmap_weights is None:
            args.weightmap_weights = list(np.ones(args.classes+1))
        if len(args.weightmap_weights) != (args.classes+1):
            parser.error(f"--classes ({args.classes}+1) needs to match --weightmap_weights number of arguments "
                         f"({len(args.weightmap_weights)})")

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

    # Train model
    run_name = utils.unique_path(path_models, args.model_name + '_{:02d}.pth').stem

    # Get CNN (double encoder U-Net)
    train_configs = {'architecture': ("MCU",
                                      args.pool_method,
                                      args.act_fun,
                                      args.norm_method,
                                      args.filters,
                                      args.channels_in,
                                      args.classes),
                     'batch_size': args.batch_size,
                     'loss': args.loss,
                     'num_gpus': num_gpus,
                     'optimizer': args.optimizer,
                     'run_name': run_name,
                     'max_epochs': args.max_epochs,
                     'upsample': args.upsample,
                     'loss_fraction_weights': args.loss_fraction_weights,
                     'weightmap_weights': args.weightmap_weights
                     }

    net = build_unet(act_fun=train_configs['architecture'][2],
                     pool_method=train_configs['architecture'][1],
                     normalization=train_configs['architecture'][3],
                     device=device,
                     num_gpus=num_gpus,
                     ch_in=train_configs['architecture'][5],
                     ch_out=train_configs['architecture'][6],
                     filters=train_configs['architecture'][4])

    # Load training and validation set
    data_transforms = augmentors(min_value=0, max_value=255)
    train_configs['data_transforms'] = str(data_transforms)
    datasets = {x: ConicDataset(root_dir=path_train_data,
                                mode=x,
                                transform=data_transforms[x])
                for x in ['train', 'val']}

    if not train_configs['max_epochs']:  # Get number of training epochs depending on dataset size if not given
        train_configs['max_epochs'] = get_max_epochs(len(datasets['train']) + len(datasets['val']))

    # Train model
    best_loss = train(net=net, datasets=datasets, configs=train_configs, device=device, path_models=path_models)

    # Fine-tune with cosine annealing for Ranger models (does not really work and the challenge model is a model
    # without cosine annealing.
    # if train_configs['optimizer'] == 'ranger':
    #     net = build_unet(act_fun=train_configs['architecture'][2],
    #                      pool_method=train_configs['architecture'][1],
    #                      normalization=train_configs['architecture'][3],
    #                      device=device,
    #                      num_gpus=num_gpus,
    #                      ch_in=train_configs['architecture'][5],
    #                      ch_out=train_configs['architecture'][6],
    #                      filters=train_configs['architecture'][4])
    #
    #     # Get best weights as starting point
    #     net = get_weights(net=net, weights=str(path_models / '{}.pth'.format(run_name)), num_gpus=num_gpus,
    #                       device=device)
    #     # Train further
    #     if train_configs['max_epochs'] >= 10 and args.train_split < 100:  # 2nd run only works for epochs > 10
    #         _ = train(net=net, datasets=datasets, configs=train_configs, device=device, path_models=path_models,
    #                   best_loss=best_loss)

    # Write information to json-file
    utils.write_train_info(configs=train_configs, path=path_models)


if __name__ == "__main__":

    main()
