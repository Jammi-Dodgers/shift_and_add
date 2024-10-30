import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


data_dir = r"C:\Users\jamie\Desktop\SharpCap Captures\2024-08-27\Capture\16_08_57"
save_dir = r"C:\Users\jamie\Pictures\astrophotos\processed"
data_files = os.listdir(data_dir)
channel = 0

images = []
for data_file in data_files:
    
    valid_file_extentions = ".png", ".tif"
    file_extention = data_file[data_file.rfind((".")):]
    if file_extention not in valid_file_extentions:
        continue
    
    data_path = os.path.join(data_dir, data_file)
    img = Image.open(data_path)
    img = img.convert("RGB")
    
    array = np.array(img)
    images += [array]
images = np.array(images)

# %%

num_images, height, length, num_channels = images.shape

# %%

peaks = [np.unravel_index(np.argmax(image), image.shape) for image in images[:,:,:,channel]]
            
y_enlargement, x_enlargement = np.max(peaks, axis= 0) -np.min(peaks, axis= 0)
shifts = np.max(peaks, axis= 0) -peaks

# %%

for peak, image in zip(peaks, images):
    plt.imshow(image[:,:,channel])
    plt.plot(peak[1], peak[0], marker= "x")
    plt.show()

# %%

shift_and_added = np.zeros((height +y_enlargement, length +x_enlargement, num_channels))
weights = np.zeros((height +y_enlargement, length +x_enlargement))
for image, shift in zip(images, shifts):
    shift_slice = np.s_[shift[0]:height +shift[0], shift[1]:length +shift[1]]
    weights[shift_slice] += np.ones_like(image[:,:,0])
    shift_and_added[shift_slice] += image
shift_and_added /= weights[:,:,None]
shift_and_added[np.isnan(shift_and_added)] = 0.0

processed = np.copy(shift_and_added)
if processed.min() <= 0.0: processed -= processed.min()
processed *= (2**8 -1) / processed.max()

processed = processed.astype(np.uint8)

img1 = Image.fromarray(processed, mode= "RGB")
plt.imshow(img1)
plt.show()

# %%

save_file = "Sun270824.tiff"
save_path = os.path.join(save_dir, save_file)

#img1.save(save_path)
