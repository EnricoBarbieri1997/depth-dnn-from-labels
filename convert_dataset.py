import numpy as np
import mat73

def convert_images_matrice_format(images):
    return np.moveaxis(images, -1, 0)

raw_matlab_data = mat73.loadmat('./datasets/nyu_depth_v2_labeled.mat')

images = convert_images_matrice_format(raw_matlab_data['images'])
depths = convert_images_matrice_format(raw_matlab_data['depths'])
labels = convert_images_matrice_format(raw_matlab_data['labels'])

new_dataset = {
    'images': images,
    'depths': depths,
    'labels': labels
}

np.savez_compressed('./datasets/nyu_depth_v2_labeled.npz', **new_dataset)