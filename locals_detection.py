import cv2
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

min_distance = cp.int32(15)  # Predefined minimum distance between neighbor local maxima
num_peaks = 10               # Predefined number of local maxima to be detected
start_frame = 5
frames_to_observe = 5
end_frame = start_frame + frames_to_observe
objects = {}

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

#start = timer()
for k in range(start_frame,end_frame+1):

    # file_name = "./IRSTD-1k/XDU" + str(k) + ".png"
    file_name = "./data/" + str(k) + ".bmp"

    image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
    peaks = np.zeros((num_peaks,2))

    # Convert image to cupy array
    image_gpu = cp.array(image)

    # Allocate memory for storing local maxima result
    result_gpu = cp.empty_like(image_gpu, dtype=bool)

    # Invoke the kernel function
    start = timer()
    kernel_peak(image_gpu, min_distance, result_gpu)
    indices_gpu = cp.argsort(cp.where(result_gpu, image_gpu, cp.zeros(image_gpu.shape)), axis=None)[-num_peaks:]
    end = timer()
    print("Time elapsed: "+str((end-start)*1000)+" ms")

    # Copy the result back to CPU
    indices = cp.asnumpy(indices_gpu)
    for i in range(num_peaks):
        peaks[i,0] = int(indices[i] / image.shape[1])
        peaks[i,1] = indices[i] % image.shape[1]
    peaks = peaks.astype('int64')

    for item in objects.keys(): # prepare identities of objects for current frame
        objects[item] = objects[item][1:]

    for coordes in peaks:
        temp_dict = objects.copy()
        for old_coordes in temp_dict.keys():
            old_coordies = np.array(old_coordes[1:-1].rsplit(" ",1))
            old_coordies = old_coordies.astype('int64')
            if abs(coordes[0] - old_coordies[0]) < 3 and abs(coordes[1] - old_coordies[1]) < 3: # check if the current object was detected in the previous frame
                temp = np.concatenate((temp_dict[old_coordes],np.ones((1,1)))) # update the identity of existing object with 1
                if np.sum(temp) == frames_to_observe:
                    print("AN OBJECT DETECTED at",old_coordes," coordinates!!!")
                    print("Frame number is:"+str(k))
                    #end = timer()
                    #print("Time elapsed: "+str((end-start)*1000)+" ms")
                    
                    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
                    ax = axes.ravel()
                    ax[0].imshow(img_h, cmap=plt.cm.gray)
                    ax[0].axis('off')
                    ax[0].set_title('Original')

                    ax[1].imshow(img_h, cmap=plt.cm.gray)
                    ax[1].autoscale(False)
                    ax[1].plot(peaks[:, 1],peaks[:, 0], 'r.')
                    ax[1].axis('off')
                    ax[1].set_title('Local Peaks')

                    fig.tight_layout()
                    plt.show()
                    exit()

                objects.pop(old_coordes)
                objects.update({str(coordes):temp})

        if objects.get(str(coordes)) is None: # if current object was not detected previously, initiate identity
            objects.update({str(coordes):np.zeros((frames_to_observe,1))})
    
    for item in objects.keys():
        if len(objects[item]) == frames_to_observe-1: # update the identity of existing object with 0
            temp = np.concatenate((objects[item],np.zeros((1,1))))
            objects.update({item:temp})

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
    #plt.close()
    plt.show()

#end = timer()
#print("Time elapsed: "+str((end-start)*1000)+" ms")