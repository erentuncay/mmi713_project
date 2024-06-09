import cv2
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

min_distance = cp.int32(15)  # Predefined minimum distance between neighbor local maxima
#min_distance = cp.int32(5)
#min_distance = cp.int32(30)

# Load the image using OpenCV
file_name = "./IRSTD-1k/XDU1.png"   # 512x512 size
# file_name = "./data/25.bmp"       # 256x256 size

image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)

# Convert image to cupy array
image_gpu = cp.array(image)

# Define CUDA kernel function for local maxima detection
kernel = cp.ElementwiseKernel(
    'raw T image, int32 n',
    'bool result',
    '''
    int x = i % image.shape()[1];
    int y = i / image.shape()[1];
    int startX = x-n;
    int endX = x+n;
    int startY = y-n;
    int endY = y+n;

    
    bool isMaxima = true;
    if (startX >= 0 && startY >= 0 && endX <= image.shape()[1] && endY <= image.shape()[0]){
        for (int dy = startY; dy < endY; ++dy) {
            for (int dx = startX; dx < endX; ++dx) {
                if (i != dy * image.shape()[1] + dx && image[i] <= image[dy * image.shape()[1] + dx]) {
                    isMaxima = false;
                    break;
                }
            }
            if (!isMaxima) break;
        }
    }

    else{    
        isMaxima = false;
    }
    
    result = isMaxima;

    ''',
    'local_maxima_detection'
)

# Allocate memory for storing local maxima result
result_gpu = cp.empty_like(image_gpu, dtype=bool)

# Invoke the kernel function
start = timer()
kernel(image_gpu, min_distance, result_gpu)
end = timer()
print("Time elapsed: "+str((end-start)*1000)+" ms")

# Copy the result back to CPU
result = cp.asnumpy(result_gpu)
local_maxima = np.where(result)

# display results
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(image, cmap=plt.cm.gray)
ax[1].autoscale(False)
ax[1].plot(np.transpose(local_maxima)[:, 1],np.transpose(local_maxima)[:, 0], 'r.')
ax[1].axis('off')
ax[1].set_title('Local Peaks')

fig.tight_layout()

plt.show()
