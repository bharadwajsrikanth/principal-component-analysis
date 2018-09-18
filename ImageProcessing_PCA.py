import cv2
from PIL import Image
from sklearn.feature_extraction import image
import numpy as np
from matplotlib import pyplot as matlib

img = cv2.imread('clockwork-angels.jpg')
img = img.astype(np.float64)
Bchannel, Gchannel, Rchannel = cv2.split(img)

patches = image.extract_patches_2d(Rchannel, patch_size=(16, 16))

cor_mat = np.zeros((256, 256))

for index in range(1000):
    vector = np.array(patches[index]).reshape(256, 1)
    interim_mat = vector.dot(vector.T)
    cor_mat = np.add(cor_mat,interim_mat)

eigenvalues, eigenvectors = np.linalg.eig(cor_mat)

idx = np.argsort(eigenvalues)[::-1]

for i in range(64):

    eig_vec = eigenvectors[:, idx[i]]
    patch = np.reshape(eig_vec, (16, 16))

    matlib.imsave("Result{}".format(i), patch.real, cmap='Greys')


back_img = Image.new('RGB', (142, 142), "white")

y = 0
img_point = 0
for y_val in range(8):
    x = 0
    for i in range(8):

        im = Image.open("Result{}.png".format(img_point))
        img_point += 1

        back_img.paste(im, (x, y))
        x = (x+18)
    y = (y+18)


back_img.show()












