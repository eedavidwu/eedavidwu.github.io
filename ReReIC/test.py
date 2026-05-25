import numpy as np
import cv2

def calculate_psnr(img1, img2):
    """计算两张图像的 PSNR（峰值信噪比）
    Args:
        img1: 原图（numpy数组，0-255）
        img2: 重建图（numpy数组，0-255）
    Returns:
        PSNR 值（float）
    """
    # 转成 float32，确保精度
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # 完全相同则 PSNR 无穷大

    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr

# ====== 示例使用 ======
# 读入原图与重建图
orig = cv2.imread('clic32.png')  # BGR 格式
recon = cv2.imread('c3.png')

# 转为 RGB 以避免通道顺序影响（可选）
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
recon = cv2.cvtColor(recon, cv2.COLOR_BGR2RGB)

# 计算 PSNR
psnr_value = calculate_psnr(orig, recon)
print(f"PSNR = {psnr_value:.2f} dB")
