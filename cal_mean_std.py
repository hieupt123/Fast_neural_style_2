import cv2
import numpy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


def mean_color(image):
    """
    Input: image RGB
    Ouput: mean color trả về 1 vector cột
    """

    image = np.uint8(image)
    img = image.copy()
    R = img[:,:,0].sum()
    G = img[:,:,1].sum()
    B = img[:,:,2].sum()
    pixel_T = numpy.transpose(np.matrix([R,G,B]))
    mean_color = pixel_T/(img.shape[0]* img.shape[1])
    return mean_color



# def covariances(path_image):
#     image = Image.open(path_image).convert('RGB')
#     image = np.uint8(image)
#     img = np.concatenate(image, axis=0)
#     mean_color = mean(path_image)
#     sum = 0
#     for i in img:
#         pixel_T = numpy.transpose(numpy.matrix(i))
#         sum += numpy.dot((pixel_T-mean_color), numpy.transpose(pixel_T - mean_color))
#     std = sum/(image.shape[0]*image.shape[1])
#     return std

def covariances2(image):

    # chuyển sang numpy array
    image_copy = image.copy()
    image_copy = np.float32(image_copy)
    img = np.concatenate(image_copy, axis=0)

    m = mean_color(image)
    # chuyển mean phù hợp với số chiều của image để tính pixel-mean
    m = np.repeat(np.transpose(m), repeats=img.shape[0], axis=0)
    img = img - m
    # print(img.shape)
    # print(img)

    img2 = np.transpose(img)
    # print(img2.shape)
    # print(img2)
    # print(np.matrix(img2[0]))
    # print(np.transpose(np.matrix(img[:, 0])))

    std = np.dot(np.matrix(img2), np.matrix(img)) / (image_copy.shape[0] * image_copy.shape[1])
    return std

def style_transformer(content, style):
    Zc = covariances2(content)
    Lc = np.linalg.cholesky(Zc)
    Zs = covariances2(style)
    Ls = np.linalg.cholesky(Zs)
    Ls_1 = np.linalg.matrix_power(Ls, -1)
    A = np.dot(Lc, Ls_1)
    b = mean_color(content) - numpy.dot(A, mean_color(style))


    image = style.copy()
    image = np.uint8(image)
    image = np.concatenate(image, axis=0)

    k = 2
    j = 0

    pixel_t = np.transpose(np.matrix(image[0]))
    style_2 = numpy.transpose(numpy.dot(A, pixel_t) + b)
    for i,img in enumerate(image):
        if i>0:
            print(i)
            pixel_t = np.transpose(np.matrix(img))
            style_1 = numpy.transpose(numpy.dot(A, pixel_t) + b)
            style_2 = numpy.concatenate((style_2,style_1), axis=0)
    return style_2


def style_transformer2(content, style):
    # tính covariances và Cholesky decomposition của ảnh style và ảnh image
    Zc = covariances2(content)
    Lc = np.linalg.cholesky(Zc)

    Zs = covariances2(style)
    Ls = np.linalg.cholesky(Zs)
    Ls_1 = np.linalg.matrix_power(Ls, -1)

    A = np.dot(Lc, Ls_1)
    b = mean_color(content) - numpy.dot(A, mean_color(style))

    image = style.copy()
    image = np.uint8(image)
    h, w, _ = image.shape

    image = np.concatenate(image, axis=0)


    A_t = np.transpose(A)
    # print(A_t.shape)
    b = np.transpose(b)
    b = np.repeat(b, repeats=image.shape[0], axis=0)
    result = np.dot(image, A_t) + b
    # print(result.shape)
    # print(result)

    img2 = np.reshape(np.array(result).astype(dtype=np.uint8), (h, w, 3))

    return img2

def conver_Yuv(path_style):
    style = cv2.imread(path_style)
    style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
    style_yuv = cv2.cvtColor(style, cv2.COLOR_RGB2YUV)
    return style_yuv

def luminance_channel(path_content, path_style):
    style_yuv = conver_Yuv(path_style)
    content_yuv = conver_Yuv(path_content)
    h, w, _ = style_yuv.shape

    mean_s = mean_color(style_yuv)
    mean_c = mean_color(content_yuv)

    std_s = covariances2(style_yuv)
    std_c = covariances2(content_yuv)

    Ls = np.concatenate(style_yuv, axis=0)
    mean_s = np.repeat(np.transpose(mean_s), repeats=Ls.shape[0], axis=0)
    mean_c = np.repeat(np.transpose(mean_c), repeats=Ls.shape[0], axis=0)
    style_1 = np.dot((Ls - mean_s), np.transpose(std_s / std_c)) + mean_c
    style_1 = np.reshape(np.array(style_1).astype(dtype=np.uint8), (h, w, 3))
    return style_1

if __name__ == "__main__":
    path_style = 'New_style/weeping-woman-by-pablo-picasso.jpg'
    path_content = 'content-images/lena.jpg'
    content = Image.open(path_content).convert('RGB')
    style = Image.open(path_style).convert('RGB')
    img = style_transformer2(content, style)
    plt.imshow(img)
    plt.show()