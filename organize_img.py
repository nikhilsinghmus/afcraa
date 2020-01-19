import sys
import os
import math
import json
import numpy
import keras
import tqdm


def main():
    filename = sys.argv[1]
    scale = int(sys.argv[2])
    X, f = load_dataset(filename, scale)

    indices = f.mean(axis=1).argsort()
    X_sorted = X[indices]

    ncols = math.ceil(math.sqrt(len(X_sorted)))

    zero_pad = numpy.zeros(((ncols ** 2) - len(X_sorted), scale, scale, 3))
    X_sorted = numpy.concatenate((X_sorted, zero_pad), axis=0)

    img = keras.preprocessing.image.array_to_img(gallery(X_sorted, ncols=ncols))
    img.save("out.png")


def gallery(a, ncols):
    n, h, w, c = a.shape
    nrows = math.ceil(n/ncols)
    output = numpy.zeros((h * nrows, w * ncols, c), dtype=a.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            output[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = a[i * ncols + j]
    return output


def load_dataset(path, scale):
    images_file = "images%d.npy" % scale
    features_file = "features.npy"
    if os.path.isfile(images_file) and os.path.isfile(features_file):
        return numpy.load(images_file), numpy.load(features_file)

    features = []
    images = []

    with open(path, "r") as input_file:
        d = json.load(input_file)

    for k, v in tqdm.tqdm(d.items()):
        if os.path.isfile(k):
            img = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(k, target_size=(scale, scale)))
            images.append(img)
            features.append(v)
            
    images = (numpy.array(images).astype(numpy.float32) / 127.5) - 1.0
    features = numpy.array(features).astype(numpy.float32)

    numpy.save(images_file, images)
    numpy.save(features_file, features)

    return images, features


if __name__ == "__main__":
	main()
