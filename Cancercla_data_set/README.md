# 数据集说明

CT cube / CT slice / PET slice / radis / coord 一一对应

## 目录结构

----Cancercla_data_set<br>
--------README.md<br>
--------label.csv<br>
--------CTSlice<br>
------------fold0<br>
----------------0001_25353_CTSlice_2.npy<br>
------------fold1<br>
------------fold2<br>
------------fold3<br>
------------fold4<br>
--------PETSlice<br>
------------fold0<br>
----------------0001_25353_PETSlice_2.npy<br>
------------fold1<br>
------------fold2<br>
------------fold3<br>
------------fold4<br>
--------CTCube<br>
------------fold0<br>
----------------0001_25353_CTCube_2.npy<br>
------------fold1<br>
------------fold2<br>
------------fold3<br>
------------fold4<br>
--------PETCube<br>
------------fold0<br>
----------------0001_25353_PETCube_2.npy<br>
------------fold1<br>
------------fold2<br>
------------fold3<br>
------------fold4<br>

说明
> 1. 三个目录中文件一一对应
> 2. 文件命名 idx+ "\_" + patient_id + "\_" + 文件类型 + "\_" + type + ".npy"

##  label.csv格式

idx|patient_id|z_slice|x_pix|y_pix|ct_r_pix|cancer_type|CT_SeriesDescription|PET_SeriesDescription|origin_dir|CT_size|PET_size|CT_pixel_spacing|PET_pixel_spacing|CT_slice_path|PET_slice_path|CT_cube_path|PET_cube_path|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
0001|25353|112|210|284|15|2|WB Standard|WB 3D MAC|2010/20100204/6273/|512_512|128_128|0.925_0.925|3.67_3.67|Cancercla_data_set/CTSlice/fold0/25353_CTSlice_2.npy|Cancercla_data_set/PETSlcie/fold0/25353_PETSLice_2.npy|Cancercla_data_set/CTCube/fold0/25353_CTCube_2.npy|Cancercla_data_set/PETCube/fold0/25353_PETCube_2.npy|


说明
> 1. 癌症类型说明 0：小细胞癌 1：2：3：4： TODO
> 2.

