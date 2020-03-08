import numpy as np
import pydicom
import csv
import os
import shutil
csv_col_name = ["idx", "patient_id", "z_slice", "x_pix", "y_pix", "ct_r_pix", "cancer_type",
                "CT_SeriesDescription", "PET_SeriesDescription", "origin_dir",
                "CT_size","PET_size","CT_pixel_spacing","PET_pixel_spacing",
                "CT_slice_path", "PET_slice_path", "CT_cube_path", "PET_cube_path"]

# 提取CT图像素值（-4000，4000），CT图的像素值是由HU值表示的
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0  # 因为小于-2000是黑色
    for slice_number in range(len(slices)):
        # Hu=pixel_val*rescale_slope+rescale_intercept
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

# 将输入图像的像素值（-1024，2000）归一化到0-1之间
def normalize_hu(image):
    # MIN_BOUND = -1000.0
    # MAX_BOUND = 400.0
    MIN_BOUND = -1350.0
    MAX_BOUND = 150.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def load_CTSlice(origin_dir,CT_SeriesDescription,z_slice):
    slices = [pydicom.read_file(origin_dir + '/' + s) for s in os.listdir(origin_dir)]
    CT_slices = []
    for s in slices:
        if s.SeriesDescription == CT_SeriesDescription:
            CT_slices.append(s)
    CT_slices.sort(key=lambda x: int(x.InstanceNumber))
    pixels = get_pixels_hu(CT_slices)
    slice = pixels[z_slice-1]  # 因为从零开始计数
    slice = normalize_hu(slice)
    return slice


if __name__ == "__main__":
    origin_root_dir = "H:/dataset/graduate_data"
    des_root_dir = "H:/dataset/graduate_data/Cancercla_data_set/CTSlice"
    related_root_prefix = "Cancercla_data_set/CTSlice"
    if not os.path.exists(des_root_dir):
        os.mkdir(des_root_dir)
    else:
        shutil.rmtree(des_root_dir)
        os.mkdir(des_root_dir)
    for i in range(5):
        os.mkdir(des_root_dir+"/fold"+str(i))
    fold_flag = [0 for i in range(5)]
    des_flag_p = [0,0,0,0,0] # 一共5类，每一类

    output_dict_rows = []
    with open("label.csv") as f_r:
        reader = csv.DictReader(f_r)
        for row in reader:
            z_slice = int(row["z_slice"])
            origin_dir = row["origin_dir"]
            origin_dir = origin_root_dir + "/" + origin_dir
            CT_SeriesDescription = row["CT_SeriesDescription"]
            cancer_type = row["cancer_type"]
            idx = row["idx"]
            patient_id = row["patient_id"]
            print("processing %s" % idx)
            slice = load_CTSlice(origin_dir,CT_SeriesDescription,z_slice)
            npy_file_name = "%04d"%int(idx) + "_" + patient_id + "_CTSlice_" + cancer_type + ".npy"
            npy_file_path = des_root_dir + "/fold" + str(des_flag_p[int(cancer_type)]) + "/" + npy_file_name
            np.save(npy_file_path,slice)
            row["CT_slice_path"] = related_root_prefix + "/fold" + str(des_flag_p[int(cancer_type)]) + "/" + npy_file_name
            des_flag_p[int(cancer_type)] = (des_flag_p[int(cancer_type)] + 1) % 5
            output_dict_rows.append(row)
        with open("new_label.csv", "w", newline='') as f_w:
            w = csv.DictWriter(f_w, output_dict_rows[0].keys())
            w.writeheader()
            w.writerows(output_dict_rows)






