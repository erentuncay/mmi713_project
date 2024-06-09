import cv2
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import numpy as np


min_distance = 15   # Predefined minimum distance between neighbor local maxima
num_peaks = 10       # Predefined number of local maxima to be detected
start_frame = 5
frames_to_observe = 5
end_frame = start_frame + frames_to_observe

start = timer()
for k in range(start_frame,end_frame+1):

    # file_name = "./data/" + str(k) + ".bmp"
    file_name = "./IRSTD-1k/XDU" + str(k) + ".png"

    # Load the image using OpenCV
    img_h = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
    labels = np.zeros(img_h.shape)
    peaks = np.zeros((num_peaks,2))

    for i in range(min_distance,img_h.shape[0]-min_distance):
        for j in range(min_distance,img_h.shape[1]-min_distance):
            if img_h[i,j] == np.max(img_h[i-min_distance:i+min_distance,j-min_distance:j+min_distance]):
                if np.sum(img_h[i,j] == img_h[i-min_distance:i+min_distance,j-min_distance:j+min_distance]) == 1:
                    labels[i,j] = img_h[i,j]

    indices = np.argsort(labels, axis=None)[-num_peaks:]    

    for i in range(num_peaks):
        peaks[i,0] = int(indices[i] / img_h.shape[1])
        peaks[i,1] = indices[i] % img_h.shape[1]

    # display results
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
    plt.close()
    #plt.show()
    
end = timer()
print("Time elapsed: "+str((end-start)*1000)+" ms")