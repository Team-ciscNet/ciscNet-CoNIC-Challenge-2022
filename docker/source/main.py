
import itk
import numpy as np
from pathlib import Path
import pandas
# import tifffile as tiff
import torch

from source.cell_segmentation_dataset import ConicDataset, infer_transforms
from source.inference import inference_2d
from source.utils import print_dir, recur_find_ext, save_as_json

def run(
        input_dir: str,
        output_dir: str,
        user_data_dir: str,
    ) -> None:
    """Entry function for automatic evaluation.

    This is the function which will be called by the organizer
    docker template to trigger evaluation run. All the data
    to be evaluated will be provided in "input_dir" while
    all the results that will be measured must be saved
    under "output_dir". Participant auxiliary data is provided
    under  "user_data_dir".

    input_dir (str): Path to the directory which contains input data.
    output_dir (str): Path to the directory which will contain output data.
    user_data_dir (str): Path to the directory which contains user data. This
        data include model weights, normalization matrix etc. .

    """
    # ! DO NOT MODIFY IF YOU ARE UNCLEAR ABOUT API !
    # <<<<<<<<<<<<<<<<<<<<<<<<< 
    # ===== Header script for user checking
    print(f"INPUT_DIR: {input_dir}")
    # recursively print out all subdirs and their contents
    print_dir(input_dir)
    print("USER_DATA_DIR: ")
    # recursively print out all subdirs and their contents
    print_dir(user_data_dir)
    print(f"OUTPUT_DIR: {output_dir}")

    paths = recur_find_ext(f"{input_dir}", [".mha"])
    assert len(paths) == 1, "There should only be one image package."
    IMG_PATH = paths[0]

    # convert from .mha to .npy
    images = np.array(itk.imread(IMG_PATH))
    np.save("images.npy", images)
    # >>>>>>>>>>>>>>>>>>>>>>>>>

    # ===== Whatever you need (function calls or complete algorithm) goes here
    # Set device for using CPU or GPU
    device, num_gpus = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
        num_gpus = 1

    # Load model
    model = Path.cwd() / "model.pth"

    updownsample = False
    th_cell = 0.12
    th_seed = 0.45
    use_tta = True

    dataset = ConicDataset(root_dir=Path.cwd(), transform=infer_transforms(updownsample, min_value=0, max_value=255))
    result_size = (images.shape[0], images.shape[1], images.shape[2], 2)
    results, counts_df = inference_2d(res_size=result_size,
                                      model=model,
                                      dataset=dataset,
                                      device=device,
                                      batchsize=8,
                                      th_cell=th_cell,
                                      th_seed=th_seed,
                                      use_tta=use_tta,
                                      downsample=updownsample,
                                      num_gpus=num_gpus)

    # For segmentation, the result must be saved at
    #     - /output/<uid>.mha
    # with <uid> is can anything. However, there must be
    # only one .mha under /output.
    itk.imwrite(
        itk.image_from_array(results),
        f"{output_dir}/pred_seg.mha"
    )
    # ToDo: remove
    # np.save(f"{output_dir}/images.npy", images)
    # np.save(f"{output_dir}/pred_seg.npy", results)
    # tiff.imsave(f"{output_dir}/images.tiff", images)
    # tiff.imsave(f"{output_dir}/pred_seg.tiff", np.transpose(results, (0, 3, 1, 2)))
    # tiff.imsave(f"{output_dir}/pred_seg2.tiff", np.concatenate((results, np.zeros(shape=(len(images), 256, 256, 1), dtype=results.dtype)), axis=-1))
    # counts_df.to_csv(f"{output_dir}/counts.csv", index=False)


    # For regression, the result for counting "neutrophil",
    # "epithelial", "lymphocyte", "plasma", "eosinophil",
    # "connective" must be respectively saved at
    #     - /output/neutrophil-count.json
    #     - /output/epithelial-cell-count.json
    #     - /output/lymphocyte-count.json
    #     - /output/plasma-cell-count.json
    #     - /output/eosinophil-count.json
    #     - /output/connective-tissue-cell-count.json
    TYPE_NAMES = [
        "neutrophil",
        "epithelial-cell",
        "lymphocyte",
        "plasma-cell",
        "eosinophil",
        "connective-tissue-cell"
    ]
    for _, type_name in enumerate(TYPE_NAMES):
        cell_counts = counts_df[type_name].to_list()
        save_as_json(
            cell_counts,
            f'{output_dir}/{type_name}-count.json'
        )
