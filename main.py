# main.py
# Full RDH pipeline: embed, extract, recover, evaluate

import cv2
import numpy as np

from utils import text_to_bits, bits_to_text
from embed import embed_payload
from extract import extract_payload
from metrics import compute_metrics

def main():
    # 1. Load input image
    original = cv2.imread("data/input.png", cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError("Input image not found in data/input.png")

    original = original.astype(np.int32)

    # 2. Define payload
    payload_text = "PatientID123"
    payload_bits = text_to_bits(payload_text)

    print("Payload text:", payload_text)
    print("Payload length (bits):", len(payload_bits))

    # 3. Embed payload
    stego, bits_embedded = embed_payload(original, payload_bits)
    print("Bits embedded:", bits_embedded)

    # Save stego image
    cv2.imwrite("results/stego.png", stego.astype(np.uint8))

    # 4. Extract payload and recover image
    recovered, extracted_bits = extract_payload(stego, bits_embedded)

    # Save recovered image
    cv2.imwrite("results/recovered.png", recovered.astype(np.uint8))

    # 5. Verify reversibility
    is_lossless = np.array_equal(original, recovered)
    print("Perfect recovery:", is_lossless)

    # 6. Recover text
    recovered_text = bits_to_text(extracted_bits)
    print("Recovered text:", recovered_text)

    # 7. Compute metrics
    psnr, ssim = compute_metrics(original, stego)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")


if __name__ == "__main__":
    main()
