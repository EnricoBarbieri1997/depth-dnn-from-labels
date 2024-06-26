from matplotlib import pyplot as plt
import numpy as np

dataset = np.load('./datasets/nyu_depth_v2_labeled.npz')
images = dataset['images']
depths = dataset['depths']
labels = dataset['labels']

for i in range(len(images)):
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(images[i])
    axarr[0,1].imshow(depths[i])
    axarr[1,0].imshow(labels[i])

    axarr[1,1].set_axis_off()
    plt.show()