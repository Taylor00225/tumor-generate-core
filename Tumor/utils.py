import os
import random

import cv2
import elasticdeform
import monai.data
import monai.transforms as transforms
import numpy as np
import torch
from scipy.ndimage import geometric_transform

path = os.getenv("DemoPath")


def SynthesisTumor(volume_scan, mask_scan, tumor_type, b):
    """
    :param b:
    :param volume_scan: original ct
    :param mask_scan: liver mask, liver 1, background 0
    :param tumor_type:
    :return:
    """

    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # consider the edg , shrink the boundary to make processed image smaller
    x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
    y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
    z_start, z_end = max(0, z_start + 1), min(mask_scan.shape[2], z_end - 1)
    liver_volume = volume_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    liver_mask = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]

    liver_volume, liver_mask = get_tumor(liver_volume, liver_mask, tumor_type, b)

    # replace
    volume_scan[x_start:x_end, y_start:y_end, z_start:z_end] = torch.tensor(liver_volume)
    mask_scan[x_start:x_end, y_start:y_end, z_start:z_end] = torch.tensor(liver_mask)

    return volume_scan, mask_scan


# Step 1
def random_select(volume_scan, mask_scan, b, radius):
    mean_hu = np.mean(volume_scan, where=np.array(mask_scan, dtype=bool))
    std_hu = np.std(volume_scan, where=np.array(mask_scan, dtype=bool))
    t = mean_hu + b
    sigma_a = 0.5 + 0.025 * std_hu
    smooth_scan = transforms.GaussianSmooth(sigma_a)(volume_scan)

    vessel_mask = np.array((smooth_scan * mask_scan) > t, dtype=int)

    # collision detection
    while True:
        coordinates = np.argwhere(mask_scan == 1)  # liver coordinates
        random_index = np.random.randint(0, len(coordinates))
        xyz = coordinates[random_index].tolist()
        x_start, x_end = xyz[0] - radius, xyz[0] + radius
        y_start, y_end = xyz[1] - radius, xyz[1] + radius
        z_start, z_end = xyz[2] - radius, xyz[2] + radius

        hit_box_1 = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]
        hit_box_2 = vessel_mask[x_start:x_end, y_start:y_end, z_start:z_end]
        if len(np.argwhere((hit_box_1 * hit_box_2) >= 1)) == 0:
            break
    return xyz, vessel_mask


# Step 2 : Generate texture
def get_texture(volume_scan, mask_scan):
    mask_shape = mask_scan.shape
    mean_hu = np.mean(volume_scan, where=np.array(mask_scan, dtype=bool))
    # std_hu should be calculated with the liver area excluding vessels,  but it is unnecessary
    # it is concluded the entire liver
    std_hu = np.std(volume_scan, where=np.array(mask_scan, dtype=bool))

    mean_hu_t = np.random.uniform(30, mean_hu - 10)

    # 1
    fix = 0.9  # use to fix the difference between non-vessels and vessels
    bg = np.zeros(mask_shape, dtype=np.float32)
    texture = transforms.RandGaussianNoise(prob=1.0, mean=mean_hu_t, std=fix * std_hu)(bg)

    # test
    # transforms.SaveImage(output_dir=path + '\\output_test\\texture\\texture_0')(texture)

    # 2
    scale_factor = np.random.uniform(1.4, 1.8)
    texture = transforms.Zoom(zoom=scale_factor, mode='bicubic')(texture)
    texture = monai.data.MetaTensor(np.array(texture))
    # transforms.SaveImage(output_dir=path + '\\output_test\\texture\\texture_1')(texture)

    # 3
    sigma_b = 0.5
    texture = transforms.GaussianSmooth(sigma=sigma_b)(texture)

    # test
    # transforms.SaveImage(output_dir=path + '\\output_test\\texture\\texture_2')(texture)

    return texture


# generate the ellipsoid
def get_ellipsoid(x, y, z):
    """
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    sh = (4 * x, 4 * y, 4 * z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2 * x, 2 * y, 2 * z])  # center point

    # calculate the ellipsoid
    bboxl = np.floor(com - radii).clip(0, None).astype(int)
    bboxh = (np.ceil(com + radii) + 1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice, bboxl, bboxh))]
    roiaux = aux[tuple(map(slice, bboxl, bboxh))]
    logrid = *map(np.square, np.ogrid[tuple(
        map(slice, (bboxl - com) / radii, (bboxh - com - 1) / radii, 1j * (bboxh - bboxl)))]),
    dst = (1 - sum(logrid)).clip(0, None)
    mask = dst > roiaux
    roi[mask] = 1
    np.copyto(roiaux, dst, where=mask)

    return out


def get_fixed_geo(volume_scan, mask_scan, tumor_type, b):
    # enlarge image to avoid collision with image edge
    enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros(
        (mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z),
        dtype=np.int8)

    def generate(tumor_num, radius, ellipsoid_sigma):
        data = np.array([], dtype=dict)
        for _ in range(tumor_num):
            x = random.randint(int(0.75 * radius), int(1.25 * radius))
            y = random.randint(int(0.75 * radius), int(1.25 * radius))
            z = random.randint(int(0.75 * radius), int(1.25 * radius))

            geo = get_ellipsoid(x, y, z)

            # test
            # if _ == 0:
            #     transforms.SaveImage(output_dir=path + '\\output_test\\shape\\shape_0')(geo)

            geo = elasticdeform.deform_random_grid(geo, sigma=ellipsoid_sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=ellipsoid_sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=ellipsoid_sigma, points=3, order=0, axis=(0, 2))

            # test
            # if _ == 0:
            #     transforms.SaveImage(output_dir=path + '\\output_test\\shape\\shape_1')(geo)

            point, vessel_mask = random_select(volume_scan, mask_scan, b, radius)

            # test
            # if _ == 0:
            #     transforms.SaveImage(output_dir=path + '\\output_test\\location')(vessel_mask)

            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # d = {'point': point, 'radius': np.max([x, y, z]), 'max_axis': np.argmax([x, y, z])}
            d = {'point': point, 'radius': radius}
            data = np.append(data, d)
        return data

    tiny_radius, small_radius, medium_radius, large_radius = 4, 8, 16, 32
    if tumor_type == 'tiny':
        num_tumor = random.randint(3, 10)
        sigma = np.random.uniform(0.5, 1)
        loc_and_r = generate(num_tumor, tiny_radius, sigma)
    elif tumor_type == 'small':
        num_tumor = random.randint(3, 10)
        sigma = np.random.uniform(1, 2)
        loc_and_r = generate(num_tumor, small_radius, sigma)
    elif tumor_type == 'medium':
        num_tumor = random.randint(2, 5)
        sigma = np.random.uniform(3, 6)
        loc_and_r = generate(num_tumor, medium_radius, sigma)
    elif tumor_type == 'large':
        num_tumor = random.randint(1, 3)
        sigma = random.uniform(5, 10)
        loc_and_r = generate(num_tumor, large_radius, sigma)

    geo_mask = geo_mask[enlarge_x // 2:-enlarge_x // 2, enlarge_y // 2:-enlarge_y // 2, enlarge_z // 2:-enlarge_z // 2]
    geo_mask = (geo_mask * mask_scan) >= 1

    # test
    # transforms.SaveImage(output_dir=path + '\\output_test\\shape\\shape_2')(geo_mask)

    # blur shape
    sigma_c = np.random.uniform(0.6, 1.2)
    geo_mask = np.array(geo_mask, dtype=np.float32)
    geo_blur = transforms.GaussianSmooth(sigma_c)(geo_mask)

    # test
    # transforms.SaveImage(output_dir=path + '\\output_test\\shape\\shape_3')(geo_blur)

    return geo_blur, loc_and_r


def get_tumor(volume_scan, mask_scan, tumor_type, b):
    print("Shape generation starting.")
    geo_blur, inform = get_fixed_geo(volume_scan, mask_scan, tumor_type, b)
    print("Shape generated.")

    print("Texture generation starting.")
    texture_scan = get_texture(volume_scan, mask_scan)
    print("Texture generated.")

    print("Combining Shape and texture starting.")
    geo_blur_mask = np.where(geo_blur > 0.0, 1.0, 0.0)
    abnormally = (1 - geo_blur) * volume_scan
    abnormally_scan = abnormally + geo_blur * texture_scan
    abnormally_mask = mask_scan + geo_blur_mask  # bg: 0, liver: 1, tumor: 2
    print("Shape and texture combined.")

    # test
    # transforms.SaveImage(output_dir='\\output_test\\post\\post_0')(abnormally_scan)

    print("Post-processing starting.")
    print("-Mass effects generation starting.")
    abnormally_scan, abnormally_mask = mass_effects(abnormally_scan, abnormally_mask, inform)
    print("-Mass effects generated.")

    # test
    # transforms.SaveImage(output_dir=path + '\\output_test\\post\\post_1')(abnormally_scan)

    print("-Capsule appearance generation starting.")
    lb, ub = 0.4, 0.7
    d, sigma_d = 80, 0.8  # d = 120x
    edge = np.where(geo_blur >= lb, 1.0, 0.0) * np.where(geo_blur <= ub, 1.0, 0.0)
    edge = transforms.GaussianSmooth(sigma_d)(edge)
    # test
    # transforms.SaveImage(output_dir=path + '\\output_test\\post\\post_2')(edge)
    abnormally_scan = abnormally_scan + d * edge
    print("-Capsule appearance generated.")

    # test
    # transforms.SaveImage(output_dir=path + '\\output_test\\post\\post_3')(abnormally_scan)

    return abnormally_scan, abnormally_mask


# 2d image warping refer to Interactive Image Warping
def warping(image, mask, center, radius, intensity):
    pow_radius = radius * radius
    height = image.shape[0]
    width = image.shape[1]
    circle_mask = np.zeros(image.shape)

    cv2.circle(circle_mask, (center[0], center[1]), radius.astype(int), (255, 255, 255), -1)

    map_x = np.vstack([np.arange(width).astype(np.float32).reshape(1, -1)] * height)
    map_y = np.hstack([np.arange(height).astype(np.float32).reshape(-1, 1)] * width)

    offset_x = map_x - center[0]
    offset_y = map_y - center[1]
    pow_dis = offset_x * offset_x + offset_y * offset_y
    scale = intensity / 100.0
    scale = 1.0 - (1.0 - pow_dis / pow_radius) * scale
    x_p = center[0] + scale * offset_x
    y_p = center[1] + scale * offset_y
    x_p[x_p < 0] = 0
    y_p[y_p < 0] = 0
    x_p[x_p >= width] = width - 1
    y_p[y_p >= height] = height - 1
    np.copyto(x_p, map_x, where=circle_mask == 0)
    np.copyto(y_p, map_y, where=circle_mask == 0)
    x_p = x_p.astype(np.float32)
    y_p = y_p.astype(np.float32)
    result_scan = cv2.remap(np.array(image), x_p, y_p, interpolation=cv2.INTER_LINEAR)
    result_mask = cv2.remap(np.array(mask), x_p, y_p, interpolation=cv2.INTER_LINEAR)

    return result_scan, result_mask


def mass_effects(abnormally_scan, abnormally_mask, loc_and_r, intensity: np.float32 = 30.0):
    """
    # 单平面近似立体warping
    for d in loc_and_r:
        radius = d['radius']
        p = d['point']
        if d['max_axis'] == 0:
            for i in range(p[0] - radius, p[0] + radius + 1):
                process = abnormally_scan[i, :, :]
                mask = abnormally_mask[i, :, :]
                center = np.append(p[1], p[2])

                _radius = np.ceil(np.sqrt(radius * radius - (i - p[0]) * (i - p[0])))
                if _radius == 0:
                    continue
                effort_scan, effort_mask = warping(process, mask, center, _radius, intensity)
                abnormally_scan[i, :, :] = torch.tensor(effort_scan)
                abnormally_mask[i, :, :] = torch.tensor(effort_mask)
        elif d['max_axis'] == 1:
            for i in range(p[1] - radius, p[1] + radius + 1):
                process = abnormally_scan[:, i, :]
                mask = abnormally_mask[:, i, :]
                center = np.append(p[0], p[2])

                _radius = np.ceil(np.sqrt(radius * radius - (i - p[1]) * (i - p[1])))
                if _radius == 0:
                    continue
                effort_scan, effort_mask = warping(process, mask, center, radius, intensity)
                abnormally_scan[:, i, :] = torch.tensor(effort_scan)
                abnormally_mask[:, i, :] = torch.tensor(effort_mask)
        else:
            for i in range(p[2] - radius, p[2] + radius + 1):
                process = abnormally_scan[:, :, i]
                mask = abnormally_mask[:, :, i]
                center = np.append(p[1], p[2])

                _radius = np.ceil(np.sqrt(radius * radius - (i - p[2]) * (i - p[2])))
                if _radius == 0:
                    continue
                effort_scan, effort_mask = warping(process, mask, center, radius, intensity)
                abnormally_scan[:, :, i] = torch.tensor(effort_scan)
                abnormally_mask[:, :, i] = torch.tensor(effort_mask)
    """
    (x, y, z) = abnormally_scan.shape
    print(f"--Total {len(loc_and_r)} tumors need to be generated.")
    i = 0
    for d in loc_and_r:
        radius = 1.3 * d['radius']
        p = d['point']
        map_x, map_y, map_z = np.mgrid[0:x, 0:y, 0:z]
        dis = np.sqrt((map_x - p[0]) ** 2 + (map_y - p[1]) ** 2 + (map_z - p[2]) ** 2)
        ball_mask = dis <= radius

        def mapping(output_coords):
            if not ball_mask[output_coords[0], output_coords[1], output_coords[2]]:
                return output_coords
            d = dis[output_coords]
            k = 1.0 - ((1.0 - d / radius)**2) * intensity / 100.0
            xd = (output_coords[0] - p[0]) * k + p[0]
            yd = (output_coords[1] - p[1]) * k + p[1]
            zd = (output_coords[2] - p[2]) * k + p[2]

            return (xd, yd, zd)

        abnormally_scan = geometric_transform(abnormally_scan, mapping)
        abnormally_mask = geometric_transform(abnormally_mask, mapping)

        i = i + 1
        print(f"---Totally finished {i}")

    return abnormally_scan, abnormally_mask
