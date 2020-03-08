import numpy as np
import csv
import sys
import os
import pydicom

csv_col_name = ["idx", "patient_id", "z_slice", "x_pix", "y_pix", "ct_r_pix", "cancer_type",
                "CT_SeriesDescription", "PET_SeriesDescription", "origin_dir",
                "CT_size","PET_size","CT_pixel_spacing","PET_pixel_spacing",
                "CT_slice_path", "PET_slice_path", "CT_cube_path", "PET_cube_path"]  # len(csv_col_name) = 19

# with open("id_ct_pet_correspond(2).csv") as f_r:
#     with open("label.csv","w") as f_w:
#         title = f_r.readline()
#         lines = f_r.readlines()
#         f_w.write(",".join(csv_col_name) + "\n")
#         for i in range(len(lines)):
#             line = lines[i]
#             idx = "%04d"%(i+1)
#             print([item.strip() for item in line.strip().split(",")])
#             _,patient_id,z_slice,x_pix,y_pix,r_pix,cancer_type,CT_SeriesDescription,PET_SeriesDescription = [item.strip() for item in line.strip().split(",")]
#             f_w.write(",".join([idx,patient_id,z_slice,x_pix,y_pix,r_pix,cancer_type,CT_SeriesDescription,PET_SeriesDescription]) +","*9+ "\n")


# 创建patientid_源文件地址
data_root = "H:/dataset/graduate_data/"
patient_id_dir_dict = dict()
for year in ["2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"]:
    year_path = data_root + year
    day_list = os.listdir(year_path)
    for day in day_list:
        day_path = year_path + "/"+day
        patient_id_list = os.listdir(day_path)
        for patient_id in patient_id_list:
            abs_path = day_path + "/" + patient_id
            relative_path = year+"/" + day + "/"+ patient_id
            patient_id_dir_dict[patient_id] = relative_path
print(patient_id_dir_dict)

def load_CT_PET(origin_dir,CT_SeriesDescription,PET_SeriesDescription):
    slices = [pydicom.read_file(origin_dir + '/' + s) for s in os.listdir(origin_dir)]
    CT_slices = []
    PET_slices = []
    for s in slices:
        if s.SeriesDescription == CT_SeriesDescription:
            CT_slices.append(s)
        if s.SeriesDescription == PET_SeriesDescription:
            PET_slices.append(s)
    CT_slices.sort(key=lambda x: int(x.InstanceNumber))
    PET_slices.sort(key=lambda x: int(x.InstanceNumber))
    return CT_slices,PET_slices



with open("label.csv") as f_r:
    input_dict_rows = csv.DictReader(f_r)
    output_dict_rows = []
    # 从1开始 前闭后开
    start = 1
    end = 950
    idx = 1
    for row in input_dict_rows:
        if idx < start:
            idx += 1
            continue
        if idx == end:
            break
        else:
            idx += 1
        print("processing",idx)
        patient_id = row["patient_id"]
        # CT_SeriesDescription = row["CT_SeriesDescription"]
        # PET_SeriesDescription = row["PET_SeriesDescription"]
        # origin_dir = patient_id_dir_dict[patient_id]
        # CT_slices, PET_slices = load_CT_PET(origin_dir, CT_SeriesDescription, PET_SeriesDescription)
        # row["CT_size"] = str(CT_slices[0].Rows) + "_" + str(CT_slices[0].Columns)
        # row["PET_size"] = str(PET_slices[0].Rows) + "_" + str(PET_slices[0].Columns)
        # row["CT_pixel_spacing"] = str(CT_slices[0].PixelSpacing[0]) + "_" + str(CT_slices[0].PixelSpacing[1])
        # row["PET_pixel_spacing"] = str(PET_slices[0].PixelSpacing[0]) + "_" + str(PET_slices[0].PixelSpacing[1])
        if patient_id not in patient_id_dir_dict:
            patient_id = "0" + patient_id
            row["patient_id"] = patient_id
            origin_dir = patient_id_dir_dict[patient_id]
        origin_dir = patient_id_dir_dict[patient_id]
        row["origin_dir"] = origin_dir
        output_dict_rows.append(row)
    with open("new_label.csv", "w", newline='') as f_w:
        w = csv.DictWriter(f_w,output_dict_rows[0].keys())
        w.writeheader()
        w.writerows(output_dict_rows)
