# embed.py
# Prediction Error Expansion (PEE) embedding

import numpy as np
from predictor import predict_pixel

def embed_payload(image, bitstream):
    stego = image.copy()
    bit_index = 0
    h, w = image.shape

    for i in range(1, h):
        for j in range(1, w):

            if bit_index >= len(bitstream):
                return stego, bit_index

            pred = predict_pixel(image, i, j)
            error = image[i, j] - pred
            bit = bitstream[bit_index]

            # PEE embedding (NO SKIP)
            new_error = 2 * error + bit
            stego[i, j] = pred + new_error

            bit_index += 1

    return stego, bit_index

