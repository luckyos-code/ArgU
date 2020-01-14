import os
import rootpath
import sys
from tqdm import tqdm

ROOT_PATH = rootpath.detect()
sys.path.append(os.path.join(rootpath.detect(), 'argU'))

# Imports nach Pfad-Erweiterung
from utils.reader import read_csv, read_csv_header

RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

# Ausgabe der ersten 5 Zeilen (+ Header) in der 'args-me.csv'

head = read_csv_header(CSV_PATH)

for i, row in enumerate(read_csv(CSV_PATH, 5)):
    print(f"{i+1}.")
    for h, r in zip(head, row):
        print(f"\t{h} --> {r}")
    print()

