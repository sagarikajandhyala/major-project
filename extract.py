# extract.py
# Payload extraction and perfect image recovery for RDH

import numpy as np
from predictor import predict_pixel

def extract_payload(stego_image, num_bits):
    """
    Extract embedded bits and recover the original image
    from the stego image.
    """
    recovered = stego_image.copy()
    extracted_bits = []
    bit_count = 0

    h, w = stego_image.shape

    # Skip border pixels
    for i in range(1, h):
        for j in range(1, w):

            if bit_count >= num_bits:
                return recovered, extracted_bits

            pred = predict_pixel(recovered, i, j)
            new_error = stego_image[i, j] - pred

            # Extract bit and original error
            bit = new_error % 2
            error = new_error // 2

            # Recover original pixel
            original_pixel = pred + error

            recovered[i, j] = original_pixel
            extracted_bits.append(bit)
            bit_count += 1

    return recovered, extracted_bits
