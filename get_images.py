import sys
import os
import wget
import zipfile
import json

feature_encoding = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11, "major": 0, "minor": 1}

def main():
	fdir = "mard"
	fn = fdir + ".zip"
	if not os.path.isfile(fn) and not os.path.isdir(fdir):
		print("Downloading MARD.")
		wget.download("http://mtg.upf.edu/system/files/projectsweb/mard.zip", bar=None)

		with zipfile.ZipFile("mard.zip", "r") as zip_ref:
			zip_ref.extractall("mard")

	metadata = get_imdict("mard/mard/mard_metadata.json")

	features = ["rhythm.bpm", "rhythm.danceability", "lowlevel.average_loudness", "tonal.key_key", "tonal.key_scale"]
	output = build_images_features_dict(metadata, "mard/mard/acousticbrainz_descriptors/", features)

	with open("images_output.json", "w") as output_file:
		json.dump(output, output_file, indent=4)

def get_imdict(filepath):
	with open(filepath, "r") as input_file:
		return {d["amazon-id"]: d["imUrl"] for d in list(filter(lambda x : "imUrl" in x, map(json.loads, input_file.readlines())))}


def build_images_features_dict(metadata, features_path, features):
	output = dict()
	f_files = list(filter(lambda x : x.endswith(".json"), os.listdir(features_path)))
	for f in f_files:
		with open(check_dirpath(features_path) + f, "r") as input_file:
			d = json.load(input_file)
			fv = [get_featurevalue(d[c][f]) for c, f in (feature.split(".") for feature in features)]

			amazon_id, mbid = os.path.splitext(f)[0].split("_")
			if amazon_id in metadata:
				output[amazon_id] = {"image": metadata[amazon_id], "features": fv}
			else:
				continue

	return output

def check_dirpath(dirpath):
	return dirpath if dirpath.endswith("/") else dirpath + "/"


def get_featurevalue(f):
	if type(f) is str:
		return feature_encoding[f]
	else:
		return f

if __name__ == "__main__":
	main()