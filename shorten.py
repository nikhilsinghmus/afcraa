import sys
import os


def main():
  assert len(sys.argv) >= 3

  filename = sys.argv[1]
  new_length = int(sys.argv[2])

  with open(filename, "r") as infile:
    lines = infile.read().split(". ")

  f, ext = os.path.splitext(filename)
  with open(f + "_shortened" + ext, "w") as outfile:
    outfile.write(" ".join(lines[:new_length]))

if __name__ == "__main__":
  main()
