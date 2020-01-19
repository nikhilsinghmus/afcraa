import sys
import json


def main():
    filename = sys.argv[1]

    with open(filename, "r") as input_file:
        d = json.load(input_file)


    length = 20

    with open("labels_touse.txt", "r") as input_file:
        labels = input_file.read().splitlines()

    for k in list(d):
        v = d[k]
        v["words"] = list(filter(lambda x : x in labels, v["words"]))
        if len(v["words"]) == 0:
            del d[k]
            continue
        
        while len(v["words"]) < length:
            v["words"].append("")

        v["words"] = v["words"][:length]

    with open("padded_output.json", "w") as output_file:
        json.dump(d, output_file)

if __name__ == "__main__":
    main()
