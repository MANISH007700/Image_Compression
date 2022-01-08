# load pkges and libs #
import time
import os 
import sys 
import numpy as np 
from PIL import Image 

# implementing K Means 
s = time.time()
# define initial K centroid
def initialize_K_centroids(X, K):
    """ Choose K points from X at random """ 
    m = len(X)
    return X[np.random.choice(m, K, replace = False), :]


# define closed centroid 
def find_closed_centroid(X, centroid):
    m = len(X)
    c = np.zeros(m)
    for i in range(m):
        # find distances 
        distance = np.linalg.norm(X[i] - centroid, axis = 1)

        # assign closest cluster to c[i]
        c[i] = np.argmin(distance)
    return c  

# find distance of eacgh example with the centroid and take average and loop
def compute_means(X, idx, K):
    _, n = X.shape 
    centroid = np.zeros((K,n))
    for k in range(K):
        example = X[np.where(idx == k)]
        mean = [np.mean(col) for col in example.T]
        centroid[k] = mean 
    return centroid

# find K means 
def find_K_means(X, K, max_iter = 10):
    centroid = initialize_K_centroids(X, K)

    prev_centroid = centroid 
    for _ in range(max_iter):
        idx = find_closed_centroid(X, centroid)
        centroid = compute_means(X, idx, K)
        if (centroid == prev_centroid).all():
            return centroid
        else:
            prev_centroid = centroid

    return centroid, idx

#### Getting the Image ##### 

def load_image(path):
    """ load image from path and return a numpy array """ 

    image = Image.open(path)
    return np.asarray(image) 

image_path = "/home/lucif3r/Desktop/Image_Compression/Image_Compression/dress.jpeg"
image = load_image(image_path)
# print(image) 
# print(image.shape)
w, h, d = image.shape
X = image.reshape((w*h), d )
K = 20 # desired number of color in image 


colors, _ = find_K_means(X, K)
idx = find_closed_centroid(X, colors)


idx = np.array(idx, dtype = np.uint8)
X_reconstructed = np.array(colors[idx, :]*1, dtype = np.uint8).reshape((w,h,d))
print(X_reconstructed)
print(X_reconstructed.shape)
compressed_image = Image.fromarray(X_reconstructed)

compressed_image.save("Compressed_Image_dress.jpeg")
e = time.time()
print("Time taken: ", e-s)












