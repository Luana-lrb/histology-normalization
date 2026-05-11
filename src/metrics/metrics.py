import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import rgb2lab, deltaE_ciede2000

def metrica_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2)

def metrica_psnr(img1, img2):
    return psnr(img1, img2)

def metrica_pcc(img1, img2):
    return np.corrcoef(img1.flatten(), img2.flatten())[0,1]

def metrica_deltaE(img1, img2):
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    return deltaE_ciede2000(lab1, lab2).mean()

def metrica_qssim_approx(img1, img2):
    ssim_r = ssim(img1[:,:,0], img2[:,:,0])
    ssim_g = ssim(img1[:,:,1], img2[:,:,1])
    ssim_b = ssim(img1[:,:,2], img2[:,:,2])
    return (ssim_r + ssim_g + ssim_b) / 3

def metricas(img1, img2):
    return {
        "SSIM": metrica_ssim(img1, img2),
        "QSSIM": metrica_qssim_approx(img1, img2),
        "PSNR": metrica_psnr(img1, img2),
        "PCC": metrica_pcc(img1, img2),
        "DeltaE": metrica_deltaE(img1, img2)
    }