import torch
import torchvision.transforms.functional as TF
import math
import numpy as np

def transform_point_affine(pt, angle, translate, scale, img_size=(192, 192)):
    # PyTorch TF.affine:
    # center is (W / 2, H / 2) -> for (192, 192), center is (96.0, 96.0)
    # The forward mapping for coordinates:
    # When an image is transformed by affine(angle, translate, scale),
    # the coordinate transformation is:
    # rot_mat = [ [scale * cos(a), -scale * sin(a)], [scale * sin(a), scale * cos(a)] ]
    # p_centered = p - center
    # p_rot = rot_mat @ p_centered
    # p_out = p_rot + center + translate
    # Note: angle in TF.affine is in degrees, counter-clockwise.
    # In image coordinates (where y is down), counter-clockwise rotation has y' = -x*sin(a) + y*cos(a).
    W, H = img_size
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0  # or W / 2.0
    rad = -math.radians(angle) # inverted because y points down
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    x, y = pt
    # Shift to center
    x_c = x - cx
    y_c = y - cy
    
    # Scale and rotate
    x_rot = scale * (x_c * cos_a - y_c * sin_a)
    y_rot = scale * (x_c * sin_a + y_c * cos_a)
    
    # Shift back and translate
    x_out = x_rot + cx + translate[0]
    y_out = y_rot + cy + translate[1]
    return x_out, y_out

if __name__ == "__main__":
    # Test on a point (80.0, 100.0)
    pt = (80.0, 100.0)
    angle = 5.0
    translate = (2.0, -3.0)
    scale = 1.05
    
    # Create delta image
    t = torch.zeros((1, 1, 192, 192))
    t[0, 0, int(pt[1]), int(pt[0])] = 1.0
    t_aff = TF.affine(t, angle=angle, translate=list(translate), scale=scale, shear=0)
    
    # Argmax in t_aff
    y_max, x_max = np.unravel_index(torch.argmax(t_aff).item(), (192, 192))
    pred_x, pred_y = transform_point_affine(pt, angle, translate, scale)
    print(f"Argmax: ({x_max}, {y_max}) | Formula: ({pred_x:.2f}, {pred_y:.2f})")
    print("Difference:", abs(x_max - pred_x), abs(y_max - pred_y))
