import math
import numpy as np
import cv2
import argparse
import os
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'scoring code'
    )
    parser.add_argument(
        '--root',
        help    = 'directory with images',
        dest = 'root',
        metavar = 'ROOT',
        type    = str,
    )
    parser.add_argument(
        '--num', dest = 'num', type = int,
        default = 24, help = 'number of data to plot'
    )

    return parser.parse_args()

def load_imgs(root):
    result = list(sorted(
        os.path.join(root, x) for x in os.listdir(root)
    ))
    # print(result)
    result = [image_to_array(x) for x in result]
    return result

# 이미지를 numpy 배열로 변환하는 함수 정의
def image_to_array(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')  # RGBA 이미지를 RGB로 변환 (알파 채널 제거)
    # img = img.resize((256, 256))  # 이미지 크기 조정 (원하는 크기로 조정)
    img_array = np.array(img)
    img.close()
    return img_array
def calculate_psnr(img1_array, img2_array):
    # img1 and img2 have range [0, 255]
    psnr_array = list()
    for i in range(len(img1_array)):
        img1 = img1_array[i]
        img2 = img2_array[i]

        psnr_array.append(psnr(img1, img2))
    return np.mean(psnr_array), np.std(psnr_array)


def calculate_ssim(img1_array, img2_array):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    ssim_array = list()
    for i in range(len(img1_array)):
        img1 = img1_array[i]
        img2 = img2_array[i]

        ssim_array.append(ssim(img1, img2, channel_axis=2))
    return np.mean(ssim_array), np.std(ssim_array)

def main():
    cmdargs = parse_cmdargs()
    gen_imgs = load_imgs(cmdargs.root + "/fake_b")
    real_imgs = load_imgs(cmdargs.root + "/real_b")


    ssim_mean, ssim_std = calculate_ssim(gen_imgs, real_imgs)
    print(f"SSIM mean is : {ssim_mean} and SSIM std is: {ssim_std}")

    psnr_mean, psnr_std = calculate_psnr(gen_imgs, real_imgs)
    print(f"PSNR mean is : {psnr_mean} and SSIM std is: {psnr_std}")

    # # 이미지 배열을 2차원으로 펼쳐서 데이터 준비
    # X = np.array(gen_imgs)
    # n_samples, height, width, channels = X.shape
    # # print(X.shape)
    # X = X.reshape(n_samples, height * width * channels)
    # labels_X = np.full(X.shape[0], 'Generated')

    # Y = np.array(real_imgs)
    # n_samples2, height2, width2, channels2 = Y.shape
    # Y = Y.reshape(n_samples2, height2*width2*channels2)    



if __name__ == '__main__':
    main()