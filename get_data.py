import sys
import os
import wget
import zipfile
import json
import tqdm

FEATURE_ENCODING = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11, "major": 0, "minor": 1}

def main():
	fdir = "mard"
	fn = fdir + ".zip"
	if not os.path.isfile(fn) and not os.path.isdir(fdir):
		print("Downloading MARD.")
		wget.download("http://mtg.upf.edu/system/files/projectsweb/mard.zip", bar=wget.bar_adaptive)

		with zipfile.ZipFile("mard.zip", "r") as zip_ref:
			zip_ref.extractall()

	print("Getting reviews and metadata.")
	reviews = get_reviewdict("mard/mard_reviews.json")
	metadata = get_metadatalist("mard/mard_metadata.json")

	print("Building dictionary with features and reviews.")
	features = ["rhythm.bpm", "rhythm.danceability", "lowlevel.average_loudness", "tonal.key_key", "tonal.key_scale"]
	output = build_reviews_features_dict(reviews, "mard/acousticbrainz_descriptors/", features)

	outdir = "output/"
	if not os.path.isdir(outdir):
		os.mkdir(outdir)

	print("Writing output to features_reviews.json.")
	with open(os.path.join(outdir, "features_reviews.json"), "w") as output_file:
		json.dump(output, output_file, indent=4)

	reviews_text = get_textcorpus(reviews)

	print("Writing reviews text output to reviews.txt.")
	with open(os.path.join(outdir, "reviews.txt"), "w") as output_file:
		output_file.writelines(reviews_text)


def get_metadatalist(filepath):
	with open(filepath, "r") as input_file:
		return list(filter(lambda x : "amazon-id" in x and "artist-mbid" in x, map(json.loads, input_file.readlines())))


def get_reviewdict(filepath):
	with open(filepath, "r") as input_file:
		return {d["amazon-id"]: d["reviewText"] for d in list(map(json.loads, input_file.readlines()))}


def get_textcorpus(reviews):
	return list(reviews.values())


def build_reviews_features_dict(reviews, features_path, features):
	output = dict()
	f_files = list(filter(lambda x : x.endswith(".json"), os.listdir(features_path)))
	for f in tqdm.tqdm(f_files):
		with open(check_dirpath(features_path) + f, "r") as input_file:
			d = json.load(input_file)
			fv = [get_featurevalue(d[c][f]) for c, f in (feature.split(".") for feature in features)]

			amazon_id, mbid = os.path.splitext(f)[0].split("_")
			if amazon_id in reviews:
				output[amazon_id] = {"review": reviews[amazon_id], "features": fv}
			else:
				continue

	return output


def check_dirpath(dirpath):
	return dirpath if dirpath.endswith("/") else dirpath + "/"


def get_featurevalue(f):
	if type(f) is str:
		return FEATURE_ENCODING[f]
	else:
		return f

if __name__ == "__main__":
	main()
