import sys
import os
import json
import numpy
import keras

def main():
    output_dir = "output/"
    data_filename, tokens_filename = [output_dir + "classifier_data.json", output_dir + "token_ids.txt"]

    with open(data_filename, "r") as input_file:
        d = json.load(input_file)

    tokens = numpy.loadtxt(tokens_filename, delimiter=" ")

    x, y = format_data(d)
    X_train, X_test = split_dataset(x)
    Y_train, Y_test = split_dataset(y)

    model = keras.Sequential()
    model.add(keras.layers.Dense(X_train.shape[0], activation="relu", input_dim=X_train.shape[1]))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(Y_train.shape[1], activation="sigmoid"))

    optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    model.fit(X_train, Y_train, epochs=40, batch_size=1000)
    score = model.predict(X_test, batch_size=1000)

    model.save(output_dir + "model.h5")


def format_data(data_dict):
    x = numpy.array([v["features"] for v in data_dict.values()])
    y = numpy.array([v["classes"] for v in data_dict.values()], dtype=numpy.int)
    return x, y


def split_dataset(dataset, p=0.7):
    sp = int(p * len(dataset))
    return dataset[:sp], dataset[sp:]


if __name__ == "__main__":
    main()
