import cv2
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

min_distance = cp.int32(15)  # Predefined minimum distance between neighbor local maxima
num_peaks = 10      # Predefined number of local maxima to be detected
start_frame = 5
frames_to_observe = 5
end_frame = start_frame + frames_to_observe

# Define CUDA kernel function for local maxima detection
kernel_peak = cp.ElementwiseKernel(
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

start = timer()
for k in range(start_frame,end_frame+1):

    # file_name = "./data/" + str(k) + ".bmp"
    file_name = "./IRSTD-1k/XDU" + str(k) + ".png"
    image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
    peaks = np.zeros((num_peaks,2))

    # Convert image to cupy array
    image_gpu = cp.array(image)

    # Allocate memory for storing local maxima result
    result_gpu = cp.empty_like(image_gpu, dtype=bool)

    # Invoke the kernel function
    kernel_peak(image_gpu, min_distance, result_gpu)
    indices_gpu = cp.argsort(cp.where(result_gpu, image_gpu, cp.zeros(image_gpu.shape)), axis=None)[-num_peaks:]
    
    # Copy the result back to CPU
    indices = cp.asnumpy(indices_gpu)

    # Post-process the results as needed
    for i in range(num_peaks):
        peaks[i,0] = int(indices[i] / image.shape[1])
        peaks[i,1] = indices[i] % image.shape[1]

    # display results
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(image, cmap=plt.cm.gray)
    ax[1].autoscale(False)
    ax[1].plot(peaks[:, 1],peaks[:, 0], 'r.')
    ax[1].axis('off')
    ax[1].set_title('Local Peaks')

    fig.tight_layout()
    plt.close()
    #plt.show()
    
end = timer()
print("Time elapsed: "+str((end-start)*1000)+" ms")