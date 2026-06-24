import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from scipy import signal as spsig, optimize as spopt
plt.rcParams.update({'figure.figsize': [12.0, 9.0], 'font.size': 24.0,
                     'ytick.left': False, 'ytick.labelleft': False, 'xtick.bottom': False, 'xtick.labelbottom': False,
                     'image.interpolation': 'none', 'figure.facecolor': 'none'})


data_dir = r"/home/jamie/Pictures/astrophotos/2026-06-23/moon/21h58m48s"
save_dir = r"/home/jamie/Pictures/astrophotos/processed"
save_file = "Moon230626.tiff"
data_files = os.listdir(data_dir)
reference_image = 0
tracking_method= "crosscor" #"circle" #"crosscor" #"peak"
interpolation= 2.0
deconvolution= 0.0 # set to 0 to turn off

images = []
for data_file in data_files: # open all photos. High memory use.
    
    valid_file_extentions = ".png", ".tif"
    file_extention = data_file[data_file.rfind((".")):]
    if file_extention not in valid_file_extentions:
        continue
    
    data_path = os.path.join(data_dir, data_file)
    img = Image.open(data_path)
    
    array = np.array(img)
    images += [array]
images = np.array(images)

plt.figure()
img0 = Image.fromarray(images[reference_image])
plt.imshow(img0)
plt.show()

# %% RESCALE IMAGES USING INTERPOLATION

images = [cv2.resize(image, None, fx=interpolation, fy=interpolation, interpolation = cv2.INTER_CUBIC)
          for n, image
          in enumerate(images)]
images = np.array(images) # consider using dtype= np.float64 but be careful because it uses a lot of memory and doesn't work with cv2.threshold

num_images, height, length, num_channels = images.shape # doesn't work for grayscale in which case it is only dim 3.

# %% FIND DRIFTS IN PHOTOS

match tracking_method:
    case "peak":
        peaks = [np.unravel_index(np.argmax(image), image.shape) for image in images[:,:,:,0]]
        peaks = np.array(peaks)

    case "crosscor":
        peaks = []
        for image in images[:,:,:,0]:
            correlation = spsig.correlate(image.astype(float), images[reference_image,:,:,0].astype(float), mode="full", method= "fft")
            peaks += [np.unravel_index(np.argmax(correlation), correlation.shape)]
        peaks = np.array(peaks)
        print(peaks[reference_image])
    case "circle":
        peaks= np.ndarray(shape= (0,2), dtype= int)
        for image in images[:,:,:,0]:
            
            # CONSIDER BLURRING THE IMAGE IF IT IS LOOKING AT THE WRONG FEATURE
            threshold, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU) #image, threshold, max, type (int)
            contours, hierarchy = cv2.findContours(mask, 1, 2) #binary image, mode, method
            biggest_contour_idx = np.argmax([len(con) for con in contours])
            contour = contours[biggest_contour_idx] # I think the last one refers to the biggest shape.

            def radius_residuals(args):
                x0, y0 = args
                r = np.sqrt((contour[:,:,0]-x0)**2 + (contour[:,:,1]-y0)**2)
                return r.flatten() - r.mean()
            
            #(x,y), cov = spopt.leastsq(radius_residuals, x0= [np.mean(contour[:,:,0]), np.mean(contour[:,:,1])])
            
            (x,y),radius = cv2.minEnclosingCircle(contour) # THIS IS BETTER FOR CRESCENTS
            
            peaks = np.concatenate((peaks, [[round(y),round(x)]]), axis= 0)
            
            #fig, axs = plt.subplots() # Ideally, I should be updating the figure instead of making new figures.
            #axs.imshow(image, cmap= "Greys_r")
            #axs.plot(*contour[:,0,:].T, marker= ".", markersize= 16, color= "tab:red")
            #axs.plot(peaks[:,1], peaks[:,0], marker= "o", markersize= 16, markevery= [-1]) #this line should be approximately straight. If it isn't then you are likely to get ghosting
            #plt.show()

y_enlargement, x_enlargement = np.max(peaks, axis= 0) -np.min(peaks, axis= 0)
shifts = np.max(peaks, axis= 0) -peaks

fig, axs = plt.subplots()
axs.imshow(np.mean(images[reference_image], axis= 2), cmap= "Greys_r")
axs.plot(peaks[:,1], peaks[:,0], linestyle= "none", marker= ".", markersize= 3) #this line should be approximately straight. If it isn't then you are likely to get ghosting
axs.tick_params(left= True, bottom= True, labelleft= True, labelbottom= True)
plt.show()

# %% DECOVOLUTION TO MAKE IT SHARPER

if deconvolution > 0:

    softening = 1e-4

    def gaussian(x, y, sigma): # gaussian as a first assumption. An airy disk is probably worth trying.
        normalisation = 2*np.pi*sigma**2
        exponent = -1/2 *(x**2 +y**2) *sigma**-2
        return np.exp(exponent) /normalisation

    xfreq = np.fft.fftfreq(length, d= 1/length)
    yfreq = np.fft.fftfreq(height, d= 1/height)
    xgrid, ygrid = np.meshgrid(xfreq, yfreq)
    kernel = gaussian(xgrid, ygrid, sigma= deconvolution)
    kernel_FFT = np.fft.fft2(kernel, norm= "ortho")
    kernel_FFT = np.where(np.abs(kernel_FFT) <= softening, softening *np.exp(1j*np.angle(kernel_FFT)), kernel_FFT) # avoid div 0 errors
    kernel = np.fft.ifft2(kernel_FFT, norm= "ortho")

    for n, image in tqdm(enumerate(images)):
        deconvolved = np.array(image, dtype=np.complex128)
        deconvolved_FFT = np.fft.fft2(deconvolved, axes= (0, 1), norm= "ortho")
        deconvolved_FFT /= kernel_FFT[:,:,None] # not sure how to correctly normalise this.
        deconvolved = np.fft.ifft2(deconvolved_FFT, axes= (0, 1), norm= "ortho")

        deconvolved = deconvolved.real
        #if deconvolved.min() <= 0.0: deconvolved -= deconvolved.min()
        #deconvolved *= (2**8 -1) / deconvolved.max()
        images[n] = deconvolved # be careful about the type of images. We need to normalise if it is int8. float64 requires a lot of memory

    print(images.dtype)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.real(kernel))
    axs[1].imshow(np.mean(images[reference_image], axis= 2))
    plt.show()

# %% AVERAGE PHOTOS

shift_and_added = np.zeros((height +y_enlargement, length +x_enlargement, num_channels))
weights = np.zeros((height +y_enlargement, length +x_enlargement))
for image, shift in zip(images, shifts):
    shift_slice = np.s_[shift[0]:height +shift[0], shift[1]:length +shift[1]]
    weights[shift_slice] += np.ones_like(image[:,:,0])
    shift_and_added[shift_slice] += image
data_exists = weights > 0
shift_and_added /= weights[:,:,None]
shift_and_added[~data_exists] = 0.0

# %% FORMATTING TO 8 BIT DEPTH

processed = np.copy(shift_and_added)
#print(processed[data_exists].min(), processed[data_exists].max(), processed[data_exists].mean(), np.median(processed[data_exists]))
if processed.min() <= 0.0: processed -= processed.min() #avoid underflow errors
processed *= (2**8 -1) / processed.max() # normalise to 255

processed = processed.astype(np.uint8)

img2 = Image.fromarray(processed)
plt.imshow(img2)
plt.show()

# %%

save_path = os.path.join(save_dir, save_file)
img2.save(save_path)
