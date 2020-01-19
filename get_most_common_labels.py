import sys
import json
import collections

def main():
    filename = sys.argv[1]
    with open(filename, "r") as input_file:
        d = json.load(input_file)

    l = []
    for k, v in d.items():
        l.extend(v["words"])

    result = [(item, c) for items, c in collections.Counter(l).most_common() for item in [items] * c]
    result = list(set(result))
    result.sort(key=lambda x : -x[1])
    result = list(filter(lambda x : x[1] > 20 and len(x[0]) > 2, result))

    r_out = [val[0] + "\n" for val in result]

    with open("labels_touse.txt", "w") as output_file:
        output_file.writelines(r_out)

if __name__ == "__main__":
    main()
