# metrics.py
# Image quality evaluation metrics

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(original_image, stego_image):
    """
    Compute PSNR and SSIM between original and stego images.
    """
    psnr = peak_signal_noise_ratio(original_image, stego_image, data_range=255)
    ssim = structural_similarity(original_image, stego_image, data_range=255)

    return psnr, ssim
