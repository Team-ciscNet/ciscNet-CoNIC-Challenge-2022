import copy
import json
import numpy as np
import os
import pathlib


def print_dir(root_path):
    """Print out the entire directory content."""
    for root, subdirs, files in os.walk(root_path):
        print(f"-{root}")
        for subdir in subdirs:
            print(f"--D-{subdir}")
        for filename in files:
            file_path = os.path.join(root, filename)
            print(f"--F-{file_path}")


def save_as_json(data, save_path):
    """Save data to a json file.

    The function will deepcopy the `data` and then jsonify the content
    in place. Support data types for jsonify consist of `str`, `int`, `float`,
    `bool` and their np.ndarray respectively.

    Args:
        data (dict or list): Input data to save.
        save_path (str): Output to save the json of `input`.

    """
    shadow_data = copy.deepcopy(data)

    # make a copy of source input
    def walk_list(lst):
        """Recursive walk and jsonify in place."""
        for i, v in enumerate(lst):
            if isinstance(v, dict):
                walk_dict(v)
            elif isinstance(v, list):
                walk_list(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
                walk_list(v)
            elif isinstance(v, np.generic):
                v = v.item()
            elif v is not None and not isinstance(v, (int, float, str, bool)):
                raise ValueError(f"Value type `{type(v)}` `{v}` is not jsonified.")
            lst[i] = v

    def walk_dict(dct):
        """Recursive walk and jsonify in place."""
        for k, v in dct.items():
            if isinstance(v, dict):
                walk_dict(v)
            elif isinstance(v, list):
                walk_list(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
                walk_list(v)
            elif isinstance(v, np.generic):
                v = v.item()
            elif v is not None and not isinstance(v, (int, float, str, bool)):
                raise ValueError(f"Value type `{type(v)}` `{v}` is not jsonified.")
            if not isinstance(k, (int, float, str, bool)):
                raise ValueError(f"Key type `{type(k)}` `{k}` is not jsonified.")
            dct[k] = v

    if isinstance(shadow_data, dict):
        walk_dict(shadow_data)
    elif isinstance(shadow_data, list):
        walk_list(shadow_data)
    else:
        raise ValueError(f"`data` type {type(data)} is not [dict, list].")
    with open(save_path, "w") as handle:
        json.dump(shadow_data, handle)



def recur_find_ext(root_dir, ext_list):
    """Recursively find all files in directories end with the `ext` such as `ext='.png'`.

    Args:
        root_dir (str): Root directory to grab filepaths from.
        ext_list (list): File extensions to consider.
    Returns:
        file_path_list (list): sorted list of filepaths.

    """
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext_list:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def get_nucleus_ids(img):
    """ Get nucleus ids in intensity-coded label image.

    :param img: Intensity-coded nuclei image.
        :type:
    :return: List of nucleus ids.
    """

    values = np.unique(img)
    values = values[values > 0]

    return values


def min_max_normalization(img, min_value=None, max_value=None):
    """ Minimum maximum normalization.

    :param img: Image (uint8, uint16 or int)
        :type img:
    :param min_value: minimum value for normalization, values below are clipped.
        :type min_value: int
    :param max_value: maximum value for normalization, values above are clipped.
        :type max_value: int
    :return: Normalized image (float32)
    """

    if max_value is None:
        max_value = img.max()

    if min_value is None:
        min_value = img.min()

    # Clip image to filter hot and cold pixels
    img = np.clip(img, min_value, max_value)

    # Apply min-max-normalization
    img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1

    return img.astype(np.float32)


def unique_path(directory, name_pattern):
    """ Get unique file name to save trained model.

    :param directory: Path to the model directory
        :type directory: pathlib path object.
    :param name_pattern: Pattern for the file name
        :type name_pattern: str
    :return: pathlib path
    """
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


def write_train_info(configs, path):
    """ Write training configurations into a json file.

    :param configs: Dictionary with configurations of the training process.
        :type configs: dict
    :param path: path to the directory to store the json file.
        :type path: pathlib Path object
    :return: None
    """

    with open(path / (configs['run_name'] + '.json'), 'w', encoding='utf-8') as outfile:
        json.dump(configs, outfile, ensure_ascii=False, indent=2)

    return None


def zero_pad_model_input(img, pad_val=0):
    """ Zero-pad model input to get for the model needed sizes (more intelligent padding ways could easily be
        implemented but there are sometimes cudnn errors with image sizes which work on cpu ...).

    :param img: Model input image.
        :type:
    :param pad_val: Value to pad.
        :type pad_val: int.

    :return: (zero-)padded img, [0s padded in y-direction, 0s padded in x-direction]
    """

    # Tested shapes
    tested_img_shapes = [64, 128, 256, 320, 512, 768, 1024, 1280, 1408, 1600, 1920, 2048, 2240, 2560, 3200, 4096,
                         4480, 6080, 8192]

    if len(img.shape) == 3:  # 3D image (z-dimension needs no pads)
        img = np.transpose(img, (2, 1, 0))

    # More effective padding (but may lead to cuda errors)
    # y_pads = int(np.ceil(img.shape[0] / 64) * 64) - img.shape[0]
    # x_pads = int(np.ceil(img.shape[1] / 64) * 64) - img.shape[1]

    pads = []
    for i in range(2):
        for tested_img_shape in tested_img_shapes:
            if img.shape[i] <= tested_img_shape:
                pads.append(tested_img_shape - img.shape[i])
                break

    if not pads:
        raise Exception('Image too big to pad. Use sliding windows')

    if len(img.shape) == 3:  # 3D image
        img = np.pad(img, ((pads[0], 0), (pads[1], 0), (0, 0)), mode='constant', constant_values=pad_val)
        img = np.transpose(img, (2, 1, 0))
    else:
        img = np.pad(img, ((pads[0], 0), (pads[1], 0)), mode='constant', constant_values=pad_val)

    return img, [pads[0], pads[1]]
