import sys
import os
import json
import numpy
import nltk
import tqdm
import transformers


def main():
    output_dir = "output/"
    tags, tokens = [output_dir + "features_adjectives.json", output_dir + "tokens.json"]

    if not os.path.isfile(tags):
        tag_adjectives(tags, get_dict_from_file(output_dir + "features_reviews.json"))

    if not os.path.isfile(tokens):
        d = tokenize_adjectives(tokens, get_dict_from_file(tags))
        build_dataset(d, output_dir)

def get_dict_from_file(f):
    with open(f, "r") as input_file:
        d = json.load(input_file)
        return d

def tag_adjectives(f, d):
    print("Tagging adjectives.")
    output_d = {k: {"features": v["features"], "words": get_adjectives(v["review"])} for k, v in tqdm.tqdm(d.items())}

    with open(f, "w") as output_file:
        json.dump(output_d, output_file)


def tokenize_adjectives(f, d):
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("./reviews_checkpoint")

    print("Tokenizing adjectives.")
    output_d = {k: {"features": v["features"], "tokens": tokenizer.encode(v["words"] or [""])} for k, v in tqdm.tqdm(d.items())}
    with open(f, "w") as output_file:
        json.dump(output_d, output_file)

    return output_d


def build_dataset(d, output_dir):
    tokens = get_tokenlist(d)
    numpy.savetxt(output_dir + "token_ids.txt", tokens, delimiter=" ", fmt="%d")

    print("Building dataset.")
    output_d = {k: {"features": v["features"], "classes": [int(t in v["tokens"]) for t in tokens]} for k, v in tqdm.tqdm(d.items())}
    with open(output_dir + "classifier_data.json", "w") as output_file:
        json.dump(output_d, output_file)


def get_adjectives(string):
    tokens = nltk.word_tokenize(string.lower())
    return [val[0] for val in filter(lambda x : x[1] == "JJ", nltk.pos_tag(tokens))]


def get_tokenlist(input):
    l = []
    for k, v in input.items():
        l.extend(v["tokens"])

    l = list(set(l))
    l.sort()
    return l


if __name__ == "__main__":
    main()
