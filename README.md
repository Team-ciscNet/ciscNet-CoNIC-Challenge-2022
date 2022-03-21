# ciscNet - A Single-Branch Cell Instance Segmentation and Classification Network

Nuclear segmentation, classification and quantification within Haematoxylin & Eosin stained histology images.

To reproduce the results of our 4-page paper submitted to the [IEEE ISBI 2022](https://biomedicalimaging.org/2022/), go to the branch **isbi2022**. For the post-challenge analysis with a specified train-val split, go to the branch **post-processing-analysis**.

## CoNIC Challenge 2022
Our method has been newly developed for the [CoNIC Challenge 2022](https://conic-challenge.grand-challenge.org/) (challenge description [paper](https://arxiv.org/abs/2111.14485)).
We participated as team **ciscnet**.

## Prerequisites
* [Anaconda Distribution](https://www.anaconda.com/distribution/#download-section).
* For GPU use: a CUDA capable GPU (highly recommended).

## Installation
Clone the repository and set up a virtual environment:

```
git clone https://git.scc.kit.edu/ciscnet/ciscnet-conic-2022
cd ciscnet-conic-2022
conda env create -f requirements.yml
conda activate ciscnet_conic_challenge_ve
```

## Data
* [Lizard dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/)
* [CoNIC Challenge patches of the Lizard dataset](https://drive.google.com/drive/folders/1il9jG7uA4-ebQ_lNmXbbF2eOK9uNwheb)

Currently, only the CoNIC Challenge patches of the Lizard dataset are supported. The corresponding files need to be saved in *./training_data/conic_patches*. 

## Usage
- train.py: create training data sets and train models
  - *--model_name* (default='conic_model'): Suffix for the model name.
  - *--dataset* (default='conic_patches'): Data to use for training.
  - *--act_fun* (default='mish'): Activation function.
  - *--batch_size* (default=8): Batch size.
  - *--classes* (default=6): Classes to predict.
  - *--filters* (default=[64, 1024]): Filters for U-net.
  - *--loss* (default='weighted_smooth_l1'): Loss function.
  - *--multi_gpu* (default=False): Use multiple GPUs.
  - *--norm_method* (default='gn'): Normalization layer type.
  - *--optimizer* (default='ranger'): Optimizer.
  - *--pool_method* (default='conv'): Downsampling layer type.
  - *--train_split* (default=80): Train set - val set split in %.
  - *--upsample* (default=False): Apply rescaling (factor 1.25).
  - *--channels_in* (default=3): Number of input channels.
  - *--max_epochs* (default=None): Maximum number of epochs (None: auto defined).
  - *--loss_fraction_weights* (default=None): Weights for weighting the losses of the single classes (first weight: summed up channel.").
  - *--weightmap_weights* (default=None): Weights for the foreground for each class (first weight: summed up channel.").
- eval.py: evaluate specified model for various thresholds on the validation set
  - *--model*: Model to evaluate.
  - *--dataset* (default='conic_patches'): Data to use for evaluation.
  - *--batch_size* (default=8): Batch size.
  - *--multi_gpu* (default=False): Use multiple GPUs.
  - *--save_raw_pred* (default=False): Save raw predictions.
  - *--th_cell* (default=0.12): Threshold(s) for adjusting cell size (multiple inputs possible).
  - *--th_seed* (default=0.45): Threshold(s) for seed extraction.
  - *--tta* (default=False): Use test-time augmentation.
  - *--eval_split* (default=80): Train set - val set split in % (use best same as for training).
  - *--upsample* (default=False): Apply rescaling (1.25) for inference (results are original scale).

## Challenge Submission Parameters
Stated are only non-default parameters.
  - train: 
    - --multi_gpu 
    - --batch_size 16 
    - --loss_fraction_weights 1 3 1 1 3 3 1 
    - --weightmap_weights 1 2 1 1 2 2 1 
  - eval/inference:
    - --tta

## Acknowledgments
* [https://github.com/TissueImageAnalytics/CoNIC](https://github.com/TissueImageAnalytics/CoNIC)
* [https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) (Ranger optimizer)
* [https://git.scc.kit.edu/KIT-Sch-GE/2021_segmentation](https://git.scc.kit.edu/KIT-Sch-GE/2021_segmentation) (code basis)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.