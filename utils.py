
import cv2
import numpy as np
from scipy.stats import kurtosis, skew

np.seterr(divide='ignore', invalid='ignore')


# vegetation index
def calculate_index(img):
    # Calculate vegetation indices: ExR, ExG, TGI
    R, G, B = cv2.split(img)

    normalizer = np.array(R.astype("float32") + G.astype("float32") + B.astype("float32"))

    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer

    ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    return ExG


def color_index_transformation(image):
    R, G, B = cv2.split(image)
    normalizer = np.array(R.astype("float32") + G.astype("float32") + B.astype("float32"))
    normalizer[normalizer == 0] = np.float32(1)  # avoid division by 0
    r, g, b = (R, G, B) / normalizer
    # Green Leaf Index
    GLI = np.array((2 * g - r - b) / (2 * g + r + b), dtype=np.float32)
    # Excess Green Index
    ExG = np.array(2.0 * g - r - b, dtype=np.float32)
    # Excess Red Index
    ExR = np.array(1.3 * r - g)
    # Normalized Difference Index
    NDI = 128 * (((g - r) / (g + r)) + 1)
    # Excess Green minus Excess red Index
    ExGR = (2 * g - (r + b)) - (1.3 * r - g)
    # Normalized Green Red Difference Index
    NGRDI = np.array((g - r) / (g + r), dtype=np.float32)
    # Triangular greenness index
    lambda_R = 670
    lambda_G = 550
    lambda_B = 480
    TGI = -0.5 * ((lambda_R - lambda_B) * (r - g) - (lambda_R - lambda_G) * (r - b))
    # Vegetation Index
    r[r == 0] = 0.00001  # Avoid division by zero
    b[b == 0] = 0.00001  # Avoid division by zero
    VEG = g / ((r ** 0.667) * (b ** 0.333))

    desc = [GLI, ExG, ExR, NDI, ExGR, NGRDI, TGI, VEG]
    desc_names = ['GLI', 'ExG', 'ExR', 'NDI', 'ExGR', 'NGRDI', 'TGI', 'VEG']

    return desc, desc_names


def index_distribution(image, feature_name, level_id, level_mask):
    px_roi = image[level_mask == 255]
    mn = np.nanmean(px_roi)
    md = np.nanmedian(px_roi)
    kt = kurtosis(px_roi, nan_policy='omit')
    sk = skew(px_roi, nan_policy='omit')
    p75, p251 = np.nanpercentile(px_roi, [75, 25])
    iqr = p75 - p251
    p98, p02 = np.nanpercentile(px_roi, [98, 2])
    ipr = p98 - p02
    std = np.nanstd(px_roi)
    stat_names = ["mean", "median", "kurtosis", "skewness", "intqrange", "intprange", "stddev"]
    stat_names = [level_id + "_" + feature_name + "_" + n for n in stat_names]
    return [mn, md, kt, sk, iqr, ipr, std], stat_names


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size - overlap)
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points