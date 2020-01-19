import sys
import os
import json
import numpy
import keras
import tensorflow
import tqdm


def main():
    filename = sys.argv[1]
    latent_dim = int(sys.argv[2])
    img_dim = int(sys.argv[3])
    num_epochs = int(sys.argv[4])
    images, features = load_dataset(filename, img_dim)

    d_model = build_discriminator(images[0].shape[0], features[0].shape[0])
    g_model = build_generator(latent_dim, images[0].shape[0], features[0].shape[0])

    gan = build_gan(g_model, d_model)

    train(g_model, d_model, gan, images, features, latent_dim, num_epochs, 128)


def rgb2luma(img):
    return (img[:,:,0] * .29) + (img[:,:,1] * .59) + (img[:,:,2] * .11)


def load_dataset(path, img_dim):
    images_file = "images%d.npy" % img_dim
    features_file = "features.npy"
    if os.path.isfile(images_file) and os.path.isfile(features_file):
        return numpy.load(images_file), numpy.load(features_file)

    features = []
    images = []

    with open(path, "r") as input_file:
        d = json.load(input_file)

    for k, v in tqdm.tqdm(d.items()):
        if os.path.isfile(k):
            img = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(k, target_size=(img_dim, img_dim)))
            images.append(img)
            features.append(v)
            
    images = (numpy.array(images).astype(numpy.float32) / 127.5) - 1.0
    features = numpy.array(features).astype(numpy.float32)

    numpy.save(images_file, images)
    numpy.save(features_file, features)

    return images, features


def real_samples(dataset, n_samples, features):
    i = numpy.random.randint(0, dataset.shape[0], n_samples)
    return [dataset[i], features[i]], numpy.ones((n_samples, 1))


def fake_samples(generator, latent_dim, n_samples, features):
    x_input = numpy.random.randn(n_samples, latent_dim)
    i = numpy.random.randint(0, features.shape[0], n_samples)
    f = features[i]
    f[:,:3] += numpy.random.randn(n_samples, 3)
    f[:,3] = numpy.random.randint(0, 12, n_samples)
    f[:,4] = numpy.random.randint(0, 2, n_samples)
    X = generator.predict([x_input, f])
    return [X, f], numpy.zeros((n_samples, 1))


def train(generator, discriminator, gan, dataset, features, latent_dim, n_epochs=400, n_batch=16):
    batch_perepoch = dataset.shape[0] // n_batch
    half_batch = n_batch // 2
    n_features = features[0].shape[0]

    for i in range(n_epochs):
        for j in range(batch_perepoch):
            [X_real, f_real], y_real = real_samples(dataset, half_batch, features)
            dl1, _ = discriminator.train_on_batch([X_real, f_real], y_real)
            [X_fake, f], y_fake = fake_samples(generator, latent_dim, half_batch, features)
            dl2, _ = discriminator.train_on_batch([X_fake, f], y_fake)
            X_gan = numpy.random.randn(n_batch, latent_dim)
            [_, f_gan], _ = fake_samples(generator, latent_dim, n_batch, features)
            gl = gan.train_on_batch([X_gan, f_gan], numpy.ones((n_batch, 1)))
            
            print(">%d, %d/%d, d1=%.4f, d2=%.4f, g=%.4f" % (i + 1, j + 1, batch_perepoch, dl1, dl2, gl))
            
        generator.save("color_generator.h5")
        discriminator.save("color_discriminator.h5")
        gan.save("color_gan.h5")


def build_gan(generator, discriminator):
    discriminator.trainable = False
    gen_noise, gen_features = generator.input
    gan_output = discriminator([generator.output, gen_features])
    model = keras.Model([gen_noise, gen_features], gan_output)
    optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    
    try:
        model = keras.utils.multi_gpu_model(model)
    except:
        pass

    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    return model


def build_generator(latent_dim, image_dim, n_features):
    features = keras.layers.Input(shape=(n_features,))
    d = image_dim // 4
    f = keras.layers.Dense(d * d * 3)(features)
    f = keras.layers.Reshape((d, d, 3))(f)
    latent_input = keras.layers.Input(shape=(latent_dim,))
    n_nodes = 128 * d * d
    gen = keras.layers.Dense(n_nodes)(latent_input)
    gen = keras.layers.LeakyReLU(alpha=0.2)(gen)
    gen = keras.layers.Reshape((d, d, 128))(gen)
    merge = keras.layers.Concatenate()([gen, f])
    gen = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(merge)
    gen = keras.layers.LeakyReLU(alpha=0.2)(gen)
    gen = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(gen)
    gen = keras.layers.LeakyReLU(alpha=0.2)(gen)
    output_layer = keras.layers.Conv2D(3, (d, d), activation="tanh", padding="same")(gen)
    model = keras.Model([latent_input, features], output_layer)

    try:
        model = keras.utils.multi_gpu_model(model)
    except:
        pass

    return model


def build_discriminator(image_dim, n_features):
    features = keras.layers.Input(shape=(n_features,))
    f = keras.layers.Dense(image_dim * image_dim * 3)(features)
    f = keras.layers.Reshape((image_dim, image_dim, 3))(f)
    image_input = keras.layers.Input(shape=(image_dim, image_dim, 3))
    merge = keras.layers.Concatenate()([image_input, f])
    fe = keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(merge)
    fe = keras.layers.LeakyReLU(alpha=0.2)(fe)
    fe = keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(fe)
    fe = keras.layers.LeakyReLU(alpha=0.2)(fe)
    fe = keras.layers.Flatten()(fe)
    fe = keras.layers.Dropout(0.4)(fe)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(fe)
    model = keras.Model([image_input, features], output_layer)
    optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

    try:
        model = keras.utils.multi_gpu_model(model)
    except:
        pass

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


if __name__ == "__main__":
    main()
