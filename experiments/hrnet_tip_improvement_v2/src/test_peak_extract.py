import torch
import numpy as np

def extract_peaks_and_offsets_from_gaussian_heatmap(hm_tensor, min_peak_val=0.5):
    # hm_tensor: [1, H, W]
    hm = hm_tensor.squeeze().cpu().numpy()
    H, W = hm.shape
    import scipy.ndimage as ndimage
    max_filt = ndimage.maximum_filter(hm, size=3, mode="constant", cval=0.0)
    is_peak = (hm == max_filt) & (hm >= min_peak_val)
    coords = np.argwhere(is_peak) # [[y, x], ...]
    
    peak_info = []
    for u, v in coords:
        if u <= 0 or u >= H - 1 or v <= 0 or v >= W - 1:
            continue
        # 1D Quadratic fitting along x
        val_c = float(hm[u, v])
        val_l = float(hm[u, v - 1])
        val_r = float(hm[u, v + 1])
        denom_x = 2.0 * (2.0 * val_c - val_l - val_r)
        dx = (val_r - val_l) / denom_x if abs(denom_x) > 1e-6 else 0.0
        dx = max(-0.5, min(0.5, dx))
        
        # 1D Quadratic fitting along y
        val_u = float(hm[u - 1, v])
        val_d = float(hm[u + 1, v])
        denom_y = 2.0 * (2.0 * val_c - val_u - val_d)
        dy = (val_d - val_u) / denom_y if abs(denom_y) > 1e-6 else 0.0
        dy = max(-0.5, min(0.5, dy))
        
        peak_info.append({
            "grid_x": int(v),
            "grid_y": int(u),
            "offset_x": float(dx),
            "offset_y": float(dy),
            "peak_val": float(val_c)
        })
    return peak_info

if __name__ == "__main__":
    # Test on synthetic Gaussian with known continuous center (50.35, 70.20)
    cx, cy = 50.35, 70.20
    H, W = 192, 192
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    hm = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * 1.5**2))
    hm_t = torch.from_numpy(hm).unsqueeze(0).float()
    
    peaks = extract_peaks_and_offsets_from_gaussian_heatmap(hm_t)
    print("Peaks extracted:", peaks)
    p = peaks[0]
    rec_x = p["grid_x"] + p["offset_x"]
    rec_y = p["grid_y"] + p["offset_y"]
    print(f"Original: ({cx:.2f}, {cy:.2f}) | Recovered: ({rec_x:.2f}, {rec_y:.2f})")
    print(f"Error: dx={abs(cx - rec_x):.4f}, dy={abs(cy - rec_y):.4f}")
