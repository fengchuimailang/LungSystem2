import pydicom
import os
import numpy
import matplotlib.pyplot as plot
import cv2

# 得到CT扫描切片厚度，加入到切片信息中
def load_patient(src_dir):
    # src_dir是病人CT文件夹地址
    # print(os.listdir(src_dir))
    slices = [pydicom.read_file(src_dir + '/' + s) for s in os.listdir(src_dir)]
    PET_slices = []
    CT_slices = []
    for s in slices:
        if s.SeriesDescription.strip() =="PET AC 2D 8":
            PET_slices.append(s)
        if s.SeriesDescription.strip() =="CT Atten Cor Head In 3.75thk":
            CT_slices.append(s)
    print(len(PET_slices))
    print(len(CT_slices))
    CT_slices.sort(key=lambda x: int(x.InstanceNumber))
    pixels = get_pixels_hu(CT_slices)
    image = pixels[98]
    image = normalize_hu(image)
    image = image * 255
    # plot.imshow(image,"gray")
    # plot.show()
    cv2.imshow("image", image)
    cv2.waitKey(0)

    # PET_slices.sort(key=lambda x: int(x.InstanceNumber))
    # pixels = get_pixels_hu(PET_slices)
    # image = pixels[98]
    # image = normalize_hu(image)
    # image = image * 255
    # plot.imshow(image,"gray")
    # plot.show()



    # return slices


# 提取CT图像素值（-4000，4000），CT图的像素值是由HU值表示的
def get_pixels_hu(slices):
    image = numpy.stack([s.pixel_array for s in slices])
    image = image.astype(numpy.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        # Hu=pixel_val*rescale_slope+rescale_intercept
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(numpy.float64)
            image[slice_number] = image[slice_number].astype(numpy.int16)
        image[slice_number] += numpy.int16(intercept)
    return numpy.array(image, dtype=numpy.int16)

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


if __name__ == "__main__":
    load_patient("H:/dataset/graduate_data/2008/20080821/3666")