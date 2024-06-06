import umap
import umap.umap_ as umap
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import argparse
import os
from sklearn.datasets import load_digits

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'umap visualization code'
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
    # img = img.resize((64, 64))  # 이미지 크기 조정 (원하는 크기로 조정)
    img_array = np.array(img)
    img.close()
    return img_array
def str2floatarray(df):
    x = df.loc[:, '1st component'].to_numpy().astype(float)
    y = df.loc[:, '2nd component'].to_numpy().astype(float)

    return x, y

def umap_app(cmdargs, n, d, XYZ, labels_XYZ):
    # umap 모델의 인스턴스를 만듭니다.
    embedding = umap.UMAP(n_neighbors = n, min_dist = d).fit_transform(XYZ)
    df_umap = pd.DataFrame(np.concatenate((embedding, labels_XYZ.reshape(-1, 1)), axis=1), columns = ['1st component', '2nd component', 'label'])

    groups = df_umap.groupby('label')

    gen_embedded = groups.get_group('Generated')
    real_embedded = groups.get_group('Real')
    wl_embedded = groups.get_group('WL')
    gen_x, gen_y = str2floatarray(gen_embedded)
    real_x, real_y = str2floatarray(real_embedded)
    wl_x, wl_y = str2floatarray(wl_embedded)


    plt.scatter(gen_x, gen_y, color = 'r', label = 'Generated')
    plt.scatter(real_x, real_y, color = 'b', label = 'Real')
    plt.scatter(wl_x, wl_y, color = 'g', label = 'WL')

    """MNIST 검증용 코드
    # print(groups.head(10))
    # gen_embedded = groups.get_group(0.0)
    # real_embedded = groups.get_group(1.0)
    # wl_embedded = groups.get_group(2.0)
    # gen_x, gen_y = str2floatarray(gen_embedded)
    # real_x, real_y = str2floatarray(real_embedded)
    # wl_x, wl_y = str2floatarray(wl_embedded)
    # plt.scatter(gen_x, gen_y, color = 'r', label = '0')
    # plt.scatter(real_x, real_y, color = 'b', label = '1')
    # plt.scatter(wl_x, wl_y, color = 'g', label = '2')
"""

    plt.legend()
    plt.savefig(cmdargs.root + "/umap{n}_{d}.PNG".format(n=n, d=d))
    plt.close()
def main():
    cmdargs = parse_cmdargs()
    gen_imgs = load_imgs(cmdargs.root + "/fake_b")
    real_imgs = load_imgs(cmdargs.root + "/real_b")
    wl_imgs = load_imgs(cmdargs.root + "/real_a")
    # 이미지 배열을 2차원으로 펼쳐서 데이터 준비
    X = np.array(gen_imgs)
    n_samples, height, width, channels = X.shape
    # print(X.shape)
    X = X.reshape(n_samples, height * width * channels)
    labels_X = np.full(X.shape[0], 'Generated')

    Y = np.array(real_imgs)
    n_samples2, height2, width2, channels2 = Y.shape
    Y = Y.reshape(n_samples2, height2*width2*channels2)    
    labels_Y = np.full(Y.shape[0], 'Real')

    Z = np.array(wl_imgs)
    n_samples3, height3, width3, channels3 = Z.shape
    Z = Z.reshape(n_samples3, height3*width3*channels3)
    labels_Z = np.full(Z.shape[0], 'WL')

    XYZ = np.concatenate((X, Y, Z), axis = 0)
    labels_XYZ = np.concatenate((labels_X, labels_Y, labels_Z))
    # 축소한 차원의 수를 정합니다.
    # digits = load_digits()
    # XYZ = digits.data
    # labels_XYZ = digits.target
    # umap_app(cmdargs, 5, 0.3, XYZ, labels_XYZ)
    for n in (2, 3, 5, 10, 15):
        for d in (0.0, 0.1, 0.5, 0.9):
            umap_app(cmdargs, n, d, XYZ, labels_XYZ)

if __name__ == '__main__':
    main()


