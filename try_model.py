import sys
import subprocess
import numpy
import keras
import essentia.standard
import transformers

FEATURE_ENCODING = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "major": 0, "minor": 1}

def main():
    tokens_file = sys.argv[1]
    audio_file = sys.argv[2]

    tokens = numpy.loadtxt(tokens_file, delimiter=" ")
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("./reviews_checkpoint/")

    model = keras.models.load_model("model.h5")

    features = ["rhythm.bpm", "rhythm.danceability", "lowLevel.average_loudness", "tonal.key_key", "tonal.key_scale"]
    loader = essentia.standard.MonoLoader(filename=audio_file)
    extractor = essentia.standard.Extractor()
    y = loader()
    fd = extractor(y)
    danceability = essentia.standard.Danceability()
    dv, dfa = danceability(y)
    fd.set("rhythm.danceability", dv)

    fv = numpy.array([[get_featurevalue(fd[f]) for f in features]])
    out = (model.predict(fv)[0] > 0.05).astype(numpy.int)
    indices = numpy.where(out)[0]
    indices = numpy.extract(indices != (len(tokens) - 1), indices)
    tokens_out = tokens[indices]

    words = [tokenizer.decode([t]) for t in tokens_out]
    prompt = ", ".join(words) + "."
    print(prompt)

    subprocess.Popen(["python3", "run_generation.py", "--model_type", "gpt2", "--model_name_or_path", "./reviews_checkpoint/", "--length", "400", "--prompt", prompt], stdout=sys.stdout, stderr=sys.stderr)


def get_featurevalue(f):
    if type(f) is str:
        return FEATURE_ENCODING[f]
    else:
        return f


if __name__ == "__main__":
    main()