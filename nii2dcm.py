import os
import time

import SimpleITK as sitk
import numpy as np

path = os.getenv("DemoPath")
path_output = path + r"\output"
path_output_dcm = path + r"\output_dcm"

iTr = os.listdir((path_output + "\\imagesTr"))
lTr = os.listdir((path_output + "\\labelsTr"))

pixel_dtype = np.float32

print("Loaded images:")
print(iTr)

print("Loaded labels:")
print(lTr)

writer = sitk.ImageFileWriter()
writer.KeepOriginalImageUIDOn()


def writeSlices(series_tag, new_img, out_dir, i):
    image_slice = new_img[:, :, i]

    # Tags shared by the series.
    list(
        map(
            lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]),
            series_tag,
        )
    )

    # Slice specific tags.
    #   Instance Creation Date
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
    #   Instance Creation Time
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))

    # Setting the type to CT so that the slice location is preserved and
    # the thickness is carried over.
    image_slice.SetMetaData("0008|0060", "CT")

    # (0020, 0032) image position patient determines the 3D spacing between
    # slices.
    #   Image Position (Patient)
    image_slice.SetMetaData(
        "0020|0032",
        "\\".join(map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))),
    )
    #   Instance Number
    image_slice.SetMetaData("0020|0013", str(i))

    # Write to the output directory and add the extension dcm, to force
    # writing in DICOM format.
    writer.SetFileName(os.path.join(out_dir, str(i) + ".dcm"))
    writer.Execute(image_slice)


for idx in range(len(iTr)):
    path_output_dcm_itr = path_output_dcm + "\\images\\" + str(idx)
    os.makedirs(path_output_dcm_itr)

    imgTr = sitk.ReadImage(path_output + "\\imagesTr\\" + iTr[idx])

    mod_time = time.strftime("%H%M%S")
    mod_date = time.strftime("%Y%m%d")

    direction = imgTr.GetDirection()
    series_tag_values = [
        ("0008|0031", mod_time),  # Series Time
        ("0008|0021", mod_date),  # Series Date
        ("0008|0008", "GENERATE"),  # Image Type
        (
            "0020|000e",
            "1.2.826.0.1.3680043.2.114514." + mod_date + ".1" + mod_time
        ),  # Series Instance UID
        (
            "0020|000D",
            "1.2.826.0.1.3680043.2.1919810." + mod_date + ".1" + mod_time
        ),  # Study Instance UID
        (
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],
                        direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        ),  # Image Orientation
        # (Patient)
        ("0008|103e", "TumorGenerate"),  # Series Description
    ]

    if pixel_dtype == np.float32:
        # If we want to write floating point values, we need to use the rescale
        # slope, "0028|1053", to select the number of digits we want to keep. We
        # also need to specify additional pixel storage and representation
        # information.
        rescale_slope = 1  # keep three digits after the decimal point
        series_tag_values = series_tag_values + [
            ("0028|1053", str(rescale_slope)),  # rescale slope
            ("0028|1052", "0"),  # rescale intercept
            ("0028|0100", "16"),  # bits allocated
            ("0028|0101", "16"),  # bits stored
            ("0028|0102", "15"),  # high bit
            ("0028|0103", "1"),
        ]  # pixel representation

    # Write slices to output directory
    list(
        map(
            lambda i: writeSlices(series_tag_values, imgTr, path_output_dcm_itr, i),
            range(imgTr.GetDepth()),
        )
    )

for idx in range(len(lTr)):
    path_output_dcm_ltr = path_output_dcm + "\\labels\\" + str(idx)
    os.makedirs(path_output_dcm_ltr)

    imgTr = sitk.ReadImage(path_output + "\\labelsTr\\" + lTr[idx])

    mod_time = time.strftime("%H%M%S")
    mod_date = time.strftime("%Y%m%d")

    direction = imgTr.GetDirection()
    series_tag_values = [
        ("0008|0031", mod_time),  # Series Time
        ("0008|0021", mod_date),  # Series Date
        ("0008|0008", "GENERATE\\LABELS"),  # Image Type
        (
            "0020|000e",
            "1.2.826.0.1.3680043.2.114514." + mod_date + ".1" + mod_time
        ),  # Series Instance UID
        (
            "0020|000D",
            "1.2.826.0.1.3680043.2.1919810." + mod_date + ".1" + mod_time
        ),  # Study Instance UID
        (
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],
                        direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        ),  # Image Orientation
        # (Patient)
        ("0008|103e", "TumorGenerateLabels"),  # Series Description
    ]

    if pixel_dtype == np.float32:
        # If we want to write floating point values, we need to use the rescale
        # slope, "0028|1053", to select the number of digits we want to keep. We
        # also need to specify additional pixel storage and representation
        # information.
        rescale_slope = 1  # keep three digits after the decimal point
        series_tag_values = series_tag_values + [
            ("0028|1053", str(rescale_slope)),  # rescale slope
            ("0028|1052", "0"),  # rescale intercept
            ("0028|0100", "16"),  # bits allocated
            ("0028|0101", "16"),  # bits stored
            ("0028|0102", "15"),  # high bit
            ("0028|0103", "1"),
        ]  # pixel representation

    # Write slices to output directory
    list(
        map(
            lambda i: writeSlices(series_tag_values, imgTr, path_output_dcm_ltr, i),
            range(imgTr.GetDepth()),
        )
    )
