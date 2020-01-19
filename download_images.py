import sys
import os
import json
import wget


def main():
    filename = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not output_dir.endswith("/"):
        output_dir += "/"
        
    with open(filename, "r") as input_file:
        d = json.load(input_file)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_d = dict()
    for k, v in d.items():
        if os.path.isfile(output_dir + os.path.basename(v["image"])):
            print("\nFile already downloaded: " + v["image"])
            continue
        print("\nDownloading " + v["image"])
        try:
            f = wget.download(v["image"], out=output_dir)
        except:
            print("Error!")

        if os.path.isfile(f):
            output_d[f] = v["features"]

    with open("img_output.json", "w") as output_file:
        json.dump(output_d, output_file)


if __name__ == "__main__":
    main()
