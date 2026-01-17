# predictor.py
# Simple pixel predictor for Reversible Data Hiding

import numpy as np

def predict_pixel(img, i, j):
    """
    Predict the pixel value at position (i, j)
    using its neighboring pixels.
    """
    top = img[i - 1, j]
    left = img[i, j - 1]
    top_left = img[i - 1, j - 1]

    # Mean predictor
    pred = (int(top) + int(left) + int(top_left)) // 3
    return pred
