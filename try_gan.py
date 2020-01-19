import sys
import json
import numpy
import keras
import essentia.standard

FEATURE_ENCODING = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "major": 0, "minor": 1}

def main():
    audio_file = sys.argv[1]

    model = keras.models.load_model("output/generator.h5")

    features = ["rhythm.bpm", "rhythm.danceability", "lowLevel.average_loudness", "tonal.key_key", "tonal.key_scale"]
    loader = essentia.standard.MonoLoader(filename=audio_file)
    extractor = essentia.standard.Extractor()
    y = loader()
    fd = extractor(y)
    danceability = essentia.standard.Danceability()
    dv, dfa = danceability(y)
    fd.set("rhythm.danceability", dv)

    with open("img_output.json", "r") as input_file:
        d = json.load(input_file)

    # i = numpy.random.randint(len(d))
    # fv = numpy.array([list(d.values())[i] for _ in range(20)])
    fv = numpy.array([[get_featurevalue(fd[f]) for f in features] for _ in range(20)])
    z = numpy.random.randn(20, model.input[0].shape[1].value)

    X = model.predict([z, fv])
    X = (X + 1) / 2

    for i in range(20):
        img = keras.preprocessing.image.array_to_img(X[i])
        img.save("output/img%d.png" % i)


def get_featurevalue(f):
    if type(f) is str:
        return FEATURE_ENCODING[f]
    else:
        return f


if __name__ == "__main__":
    main()
