import numpy as np
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import os
import config

def t_from_hue_curve(h, m, d, n, s, min_temp=config.GLOBAL_MIN_TEMP, max_temp=config.GLOBAL_MAX_TEMP, eps=1e-12):
    """
    Convert hue -> temperature using the inverse logistic:
      T = m - (1/d) * log( n / (h - s) - 1 )
    Handles numpy arrays. For pixels:
      - h <= s        : assigned `min_temp`
      - s < h < s+n   : computed with the formula (safe vectorized)
      - h >= s + n    : assigned `max_temp` if provided, otherwise NaN
    eps: numeric small value used to avoid log(0) from numerical round-off.
    """
    h = np.asarray(h, dtype=np.float64)
    T = np.full_like(h, np.nan, dtype=np.float64)

    # masks
    below_mask = h <= s
    above_mask = h >= (s + n)
    valid_mask = (h > s) & (h < (s + n))

    # handle out-of-range quickly
    if np.any(below_mask):
        T[below_mask] = min_temp

    if np.any(above_mask):
        if max_temp is None:
            T[above_mask] = np.nan
        else:
            T[above_mask] = max_temp

    # compute only for the valid mask (avoids division-by-zero / log-of-nonpositive)
    if np.any(valid_mask):
        with np.errstate(divide="ignore", invalid="ignore"):
            arg = n / (h[valid_mask] - s) - 1.0
            # numeric safety: make sure arg is strictly positive for log()
            arg = np.maximum(arg, eps)
            T[valid_mask] = m - (1.0 / d) * np.log(arg)
    return T

# --- Elaboration Loop---

# Find all image paths
hue_image_paths = glob(f"{config.DATA_PATH}\\*.jpg")

# Create output directory if it doesn't exist
if not os.path.exists(config.TEMP_PATH):
    os.makedirs(config.TEMP_PATH)

print(f"Found {len(hue_image_paths)} images to process.")

# Variables to store the last processed image for plotting (needed only if __name__ == "__main__")
last_hue_ndarray = None
last_temp_ndarray = None
last_img_name = ""

# Process each image
for i, img_path in enumerate(hue_image_paths):
    try:

        img = Image.open(img_path)
        hue_ndarray = np.array(img)
        last_hue_ndarray = hue_ndarray # Salva per il plot
        last_img_name = os.path.basename(img_path)

        if hue_ndarray.ndim == 3 and hue_ndarray.shape[2] >= 1:
            # Immagine RGB o HSV, prendi il primo canale
            hue_channel = hue_ndarray[:, :, 0]
        elif hue_ndarray.ndim == 2:
            # If grayscale image, treat as "min_temp"
            print(f"Caution: Image {last_img_name} is grayscale. Treating as 'all min_temp'.")
            hue_channel = np.zeros_like(hue_ndarray, dtype=np.float64)
        else:
            print(f"Caution: Image {last_img_name} has unexpected shape {hue_ndarray.shape}. Skipping.")
            continue


        temp_ndarray = t_from_hue_curve(
            hue_channel,
            m=config.HUE_AVG,
            d=config.HUE_SIGMA,
            n=config.HUE_NORM,
            s=config.HUE_SHIFT,
        )
        last_temp_ndarray = temp_ndarray 

        
        valid_mask = ~np.isnan(temp_ndarray)
        
        if not np.any(valid_mask):
            temp_normalized = np.zeros_like(temp_ndarray, dtype=np.uint8)
        else:
            temp_min = config.GLOBAL_MIN_TEMP
            temp_max = config.GLOBAL_MAX_TEMP
            
            temp_filled = np.where(valid_mask, temp_ndarray, temp_min)

            if temp_max - temp_min < 1e-6:
                temp_normalized = np.zeros_like(temp_filled, dtype=np.uint8)
            else:

                temp_normalized = (temp_filled - temp_min) / (temp_max - temp_min) * 255.0
                temp_normalized = np.clip(temp_normalized, 0, 255).astype(np.uint8)


        temp_image = Image.fromarray(temp_normalized, mode="L")
        temp_image.save(os.path.join(config.TEMP_PATH, f"temp_map_{i:03d}.png"))

    except Exception as e:
        print(f"Error in elaborating Image {img_path}: {e}")

print("Elaboration completed.")

# if __name__ == "__main__":
#     # Plotta l'esempio dell'ultima immagine elaborata
#     if last_temp_ndarray is not None and last_hue_ndarray is not None:
#         fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#         ax[0].imshow(last_hue_ndarray)
#         ax[0].set_title(f"Last Hue Image ({last_img_name})")
#         ax[1].imshow(last_temp_ndarray, cmap="inferno")
#         ax[1].set_title(f"Last Temperature Map ({last_img_name})")
#         plt.show()
#     else:
#         print("Nessuna immagine elaborata con successo per il plot.")