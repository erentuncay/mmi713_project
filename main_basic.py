import cv2
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import numpy as np


min_distance = 15   # Predefined minimum distance between neighbor local maxima
#min_distance = 5
#min_distance = 30

file_name = "./IRSTD-1k/XDU1.png"   # 512x512 size
# file_name = "./data/25.bmp"       # 256x256 size

# Load the image using OpenCV
img_h = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
labels = np.zeros(img_h.shape)

start = timer()
for i in range(min_distance,img_h.shape[0]-min_distance):
    for j in range(min_distance,img_h.shape[1]-min_distance):
        if img_h[i,j] == np.max(img_h[i-min_distance:i+min_distance,j-min_distance:j+min_distance]):
            if np.sum(img_h[i,j] == img_h[i-min_distance:i+min_distance,j-min_distance:j+min_distance]) == 1: # check if it is the only maximum point
                labels[i,j] = img_h[i,j]
end = timer()
print("Time elapsed: "+str((end-start)*1000)+" ms")

# display results
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img_h, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(img_h, cmap=plt.cm.gray)
ax[1].autoscale(False)
ax[1].plot(np.transpose(np.nonzero(labels))[:, 1],np.transpose(np.nonzero(labels))[:, 0], 'r.')
ax[1].axis('off')
ax[1].set_title('Local Peaks')

fig.tight_layout()

plt.show()