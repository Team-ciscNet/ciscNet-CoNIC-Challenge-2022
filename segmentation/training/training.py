import gc
import random

import numpy as np
import time
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from segmentation.training.ranger2020 import Ranger
from segmentation.training.losses import get_loss, get_weight_map

import pandas as pd


def get_max_epochs(n_samples):
    """ Get maximum amount of training epochs.

    :param n_samples: number of training samples.
        :type n_samples: int
    :return: maximum amount of training epochs
    """

    if n_samples >= 2000:
        max_epochs = 120
    elif n_samples >= 1000:
        max_epochs = 200
    elif n_samples >= 500:
        max_epochs = 240
    elif n_samples >= 200:
        max_epochs = 320
    elif n_samples >= 100:
        max_epochs = 400
    elif n_samples >= 50:
        max_epochs = 480
    else:
        max_epochs = 560

    return max_epochs


def get_weights(net, weights, device, num_gpus):
    """ Load weights into model.

    :param net: Model to load the weights into.
        :type net:
    :param weights: Path to the weights.
        :type weights: pathlib Path object
    :param device: Device to use ('cpu' or 'cuda')
        :type device:
    :param num_gpus: Amount of GPUs to use.
        :type num_gpus: int
    :return: model with loaded weights.

    """
    if num_gpus > 1:
        net.module.load_state_dict(torch.load(weights, map_location=device))
    else:
        net.load_state_dict(torch.load(weights, map_location=device))
    return net


def seed_worker(worker_id):
    """ Fix pytorch seeds on linux

    https://pytorch.org/docs/stable/notes/randomness.html
    https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/

    :param worker_id:
    :return:
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(net, datasets, configs, device, path_models, best_loss=1e4):
    """ Train the model.

    :param net: Model/Network to train.
        :type net:
    :param datasets: Dictionary containing the training and the validation data set.
        :type datasets: dict
    :param configs: Dictionary with configurations of the training process.
        :type configs: dict
    :param device: Use (multiple) GPUs or CPU.
        :type device: torch device
    :param path_models: Path to the directory to save the models.
        :type path_models: pathlib Path object
    :param best_loss: Best loss (only needed for second run to see if val loss further improves).
        :type best_loss: float

    :return: None
    """

    print('-' * 20)
    print('Train {0} on {1} images, validate on {2} images'.format(configs['run_name'],
                                                                   len(datasets['train']),
                                                                   len(datasets['val'])))

    # Data loader for training and validation set
    apply_shuffling = {'train': True, 'val': False}

    dataloader = {x: torch.utils.data.DataLoader(datasets[x],
                                                 batch_size=configs['batch_size'],
                                                 shuffle=apply_shuffling[x],
                                                 pin_memory=True,
                                                 worker_init_fn=seed_worker,
                                                 num_workers=8)
                  for x in ['train', 'val']}

    # Loss function and optimizer
    criterion = get_loss(configs['loss'])

    second_run = False
    max_epochs = configs['max_epochs']

    # Optimizer
    if configs['optimizer'] == 'adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=8e-4,
                               betas=(0.9, 0.999),
                               eps=1e-08,
                               weight_decay=0,
                               amsgrad=True)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=0.25,
                                      patience=configs['max_epochs'] // 20,
                                      verbose=True,
                                      min_lr=3e-6) 
        break_condition = 2 * configs['max_epochs'] // 20 + 5

    elif configs['optimizer'] == 'ranger':

        lr = 6e-3
        if best_loss < 1e3:  # probably second run

            second_run = True

            optimizer = Ranger(net.parameters(),
                               lr=0.09 * lr,
                               alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                               betas=(.95, 0.999), eps=1e-6, weight_decay=0,  # Adam options
                               # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                               use_gc=True, gc_conv_only=False, gc_loc=True)

            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=configs['max_epochs'] // 10,  # train at least one more epoch
                                          eta_min=3e-5,
                                          last_epoch=-1,
                                          verbose=True)
            break_condition = configs['max_epochs'] // 10 + 1
            max_epochs = configs['max_epochs'] // 10  # train at least one more epoch
        else:
            optimizer = Ranger(net.parameters(),
                               lr=lr,
                               alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                               betas=(.95, 0.999), eps=1e-6, weight_decay=0,  # Adam options
                               # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                               use_gc=True, gc_conv_only=False, gc_loc=True)
            scheduler = ReduceLROnPlateau(optimizer,
                                          mode='min',
                                          factor=0.25,
                                          patience=configs['max_epochs'] // 10,
                                          verbose=True,
                                          min_lr=0.075*lr)
            break_condition = 2 * configs['max_epochs'] // 10 + 5
    else:
        raise Exception('Optimizer not known')

    # Auxiliary variables for training process
    epochs_wo_improvement = 0
    
    # Create dataframe for losses
    train_loss = pd.DataFrame(columns=["epoch"] + list(np.arange(configs["architecture"][-1]+1)) + ["mean"])
    val_loss = pd.DataFrame(columns=["epoch"] + list(np.arange(configs["architecture"][-1]+1)) + ["mean"])
    since = time.time()

    loss_fraction_weights = configs["loss_fraction_weights"]
    weightmap_weights = configs["weightmap_weights"]

    # Training process
    for epoch in range(max_epochs):

        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, max_epochs))
        print('-' * 10)

        start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluation mode

            # create running loss dict for each output layer and initialize with zero. E.g.: {"0":0, "1":0, ..., "6":0}
            running_loss = dict(zip(np.arange(configs["architecture"][-1]+1),
                                    np.zeros_like(np.arange(configs["architecture"][-1]+1))))

            # Iterate over data
            for samples in dataloader[phase]:
                # Get img_batch and label_batch and put them on GPU if available
                img_batch = samples["image"]
                label_batch = samples["label"]
                img_batch = img_batch.to(device)
                label_batch = label_batch.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass (track history if only in train)
                with torch.set_grad_enabled(phase == 'train'):
                    pred_batch = net(img_batch)
                    loss = {}
                    for i in range(7):
                        if configs['loss'] == 'weighted_smooth_l1':
                            loss[i] = loss_fraction_weights[i] * criterion(pred_batch[:, i, None, ...],
                                                                           label_batch[:, i, None, ...],
                                                                           get_weight_map(label_batch[:, i, None, ...],
                                                                                          weightmap_weights[i]))
                        else:
                            loss[i] = loss_fraction_weights[i] * criterion(pred_batch[:, i, None, ...],
                                                                           label_batch[:, i, None, ...])

                        # Statistics
                        running_loss[i] += float(loss[i].item() * img_batch.size(0))          
                    
                    # Backward (optimize only if in training phase)
                    if phase == 'train':
                        torch.stack(list(loss.values())).mean().backward()
                        optimizer.step()

            epoch_loss = dict(zip(np.arange(configs["architecture"][-1]+1),
                                  np.zeros_like(np.arange(configs["architecture"][-1]+1))))
            for i in range(len(running_loss)):
                epoch_loss[i] = running_loss[i] / len(datasets[phase])

            # epoch_loss = [running_loss[i] / len(datasets[phase]) for i in range(len(running_loss))]

            epoch_loss["mean"] = np.mean(list(epoch_loss.values()))
            epoch_loss["epoch"] = epoch
            epoch_loss = pd.DataFrame(epoch_loss, index=[0])  # convert do dataframe to append to train_loss or val_loss
            if phase == 'train':
                train_loss = pd.concat([train_loss, epoch_loss], ignore_index=True)
                print('Mean training loss: {:.5f}'.format(train_loss.iloc[-1, train_loss.columns.get_loc('mean')]))
            else:
                val_loss = pd.concat([val_loss, epoch_loss], ignore_index=True)
                print('Mean validation loss: {:.5f}'.format(val_loss.iloc[-1, train_loss.columns.get_loc('mean')]))

                if epoch_loss.iloc[-1]["mean"] < best_loss:
                    print('Mean validation loss improved from {:.5f} to {:.5f}. Save model.'.format(
                        best_loss,
                        epoch_loss.iloc[-1]["mean"])
                    )
                    best_loss = epoch_loss.iloc[-1]["mean"]

                    # The state dict of data parallel (multi GPU) models need to get saved in a way that allows to
                    # load them also on single GPU or CPU
                    if configs['num_gpus'] > 1:
                        torch.save(net.module.state_dict(), str(path_models / (configs['run_name'] + '.pth')))
                    else:
                        torch.save(net.state_dict(), str(path_models / (configs['run_name'] + '.pth')))
                    epochs_wo_improvement = 0

                else:
                    print('Validation loss did not improve.')
                    epochs_wo_improvement += 1

                if configs['optimizer'] == 'ranger' and second_run:
                    scheduler.step()

                else:
                    scheduler.step(epoch_loss.loc[0, "mean"])

        # Epoch training time
        print('Epoch training time: {:.1f}s'.format(time.time() - start))

        # Break training if plateau is reached
        if epochs_wo_improvement == break_condition:
            print(str(epochs_wo_improvement) + ' epochs without validation loss improvement --> break')
            break

    # Total training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('-' * 20)

    # Merge train and val loss and save loss
    train_loss = train_loss.add_prefix("train_")
    train_loss.rename(columns={"train_epoch": "epoch"}, inplace=True)
    val_loss = val_loss.add_prefix("val_")
    val_loss.rename(columns={"val_epoch": "epoch"}, inplace=True)    
    loss = train_loss.merge(val_loss, on="epoch")
    
    if second_run:
        loss.to_csv(str(path_models / (configs['run_name'] + '_2nd_loss.csv')), index=False)
        configs['training_time_run_2'], configs['trained_epochs_run2'] = time_elapsed, epoch + 1
    else:
        loss.to_csv(str(path_models / (configs['run_name'] + '_loss.csv')), index=False)
        configs['training_time'], configs['trained_epochs'] = time_elapsed, epoch + 1

    # Clear memory
    del net
    gc.collect()

    return best_loss
