import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal as spsig, optimize as spopt
plt.rcParams.update({'figure.figsize': [12.0, 9.0], 'font.size': 24.0,
                     'ytick.left': False, 'ytick.labelleft': False, 'xtick.bottom': False, 'xtick.labelbottom': False,
                     'image.interpolation': 'none', 'figure.facecolor': 'none'})


data_dir = r"C:\Users\jamie\Desktop\SharpCap Captures\2024-08-27\Capture\16_08_57"
save_dir = r"C:\Users\jamie\Pictures\astrophotos\processed"
data_files = os.listdir(data_dir)
reference_image = 22
mode= "LAB" # LAB space seems to be the best because it completely seperates luminosity from colour.
tracking_method= "circle" #"circle" #"crosscor" #"peak"
interpolation= 1.0

images = []
for data_file in data_files:
    
    valid_file_extentions = ".png", ".tif"
    file_extention = data_file[data_file.rfind((".")):]
    if file_extention not in valid_file_extentions:
        continue
    
    data_path = os.path.join(data_dir, data_file)
    img = Image.open(data_path)
    img = img.convert(mode)
    
    array = np.array(img)
    images += [array]
images = np.array(images)

plt.figure()
img0 = Image.fromarray(images[reference_image], mode= mode)
plt.imshow(img0)
plt.show()

# %%

images = [cv2.resize(image, None, fx=interpolation, fy=interpolation, interpolation = cv2.INTER_CUBIC)
          for n, image
          in enumerate(images)]
images = np.array(images)

num_images, height, length, num_channels = images.shape

# %%

match tracking_method:
    case "peak":
        peaks = [np.unravel_index(np.argmax(image), image.shape) for image in images[:,:,:,0]]
    case "crosscor":
        peaks = []
        for image in images[:,:,:,0]:
            correlation = spsig.correlate(image.astype(float), images[reference_image,:,:,0].astype(float), mode="full", method= "fft")
            peaks += [np.unravel_index(np.argmax(correlation), correlation.shape)]
    case "circle":
        peaks= np.ndarray(shape= (0,2), dtype= int)
        for image in images[:,:,:,0]:
            
            # CONSIDER BLURRING THE IMAGE IF IT IS LOOKING AT THE WRONG FEATURE
            threshold, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU) #image, threshold, max, type (int)
            contours, hierarchy = cv2.findContours(mask, 1, 2) #binary image, mode, method
            contour = contours[-1] # I think the last one refers to the biggest shape.
            
            def radius_residuals(args):
                x0, y0 = args
                r = np.sqrt((contour[:,:,0]-x0)**2 + (contour[:,:,1]-y0)**2)
                return r.flatten() - r.mean()
            
            (x,y), cov = spopt.leastsq(radius_residuals, x0= [np.mean(contour[:,:,0]), np.mean(contour[:,:,1])])
            
            #(x,y),radius = cv2.minEnclosingCircle(contour) # THIS IS BETTER FOR CRESCENTS
            
            peaks = np.concatenate((peaks, [[round(y),round(x)]]), axis= 0)
            
            fig, axs = plt.subplots()
            axs.imshow(image, cmap= "Greys_r")
            axs.plot(*contour[:,0,:].T, marker= ".", markersize= 16, color= "tab:red")
            axs.plot(peaks[:,1], peaks[:,0], marker= "o", markersize= 16, markevery= [-1]) #this line should be approximately straight. If it isn't then you are likely to get ghosting
            plt.show()
            
y_enlargement, x_enlargement = np.max(peaks, axis= 0) -np.min(peaks, axis= 0)
shifts = np.max(peaks, axis= 0) -peaks

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

img1 = Image.fromarray(processed, mode= mode)
plt.imshow(img1)
plt.show()

# %%

save_file = "Sun270824.tiff"
save_path = os.path.join(save_dir, save_file)

#img1.save(save_path)
