import os

import monai.transforms as transforms
from monai.data import load_decathlon_datalist

from Tumor import TumorGenerated

path = os.getenv("DemoPath")


def _get_transform():
    tf_compose = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            TumorGenerated(keys=["image", "label"]),
            transforms.SaveImaged(keys=["image"],
                                  squeeze_end_dims=True,
                                  resample=True,
                                  separate_folder=False,
                                  output_dir=path + "\\output\\imagesTr"),
            transforms.SaveImaged(keys=["label"],
                                  squeeze_end_dims=True,
                                  resample=True,
                                  separate_folder=False,
                                  output_dir=path + "\\output\\labelsTr")
        ]

    )
    return tf_compose


if __name__ == '__main__':

    data_list_dir = path + r"\data"
    data_list_json_dir = path + r"\data\dataset.json"
    data_list = load_decathlon_datalist(data_list_file_path=data_list_json_dir,
                                        is_segmentation=True,
                                        data_list_key="training",
                                        base_dir=data_list_dir)
    new_data_list = []
    for item in data_list:
        new_item = {'image': item['image'].replace('.npy', ''), 'label': item['label'].replace('.npy', '')}
        new_data_list.append(new_item)
    print(f'Loaded {len(new_data_list)} file(s):')
    print(new_data_list)

    tf = _get_transform()

    for data in data_list:
        d = tf(data)

    print("done")
