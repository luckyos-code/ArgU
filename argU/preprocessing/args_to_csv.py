import json
import csv
import os
from tqdm import tqdm

import rootpath

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')

json_path = os.path.join(RESOURCES_PATH, 'args-me.json')
csv_path = os.path.join(RESOURCES_PATH, 'args-me.csv')

# Relevante Spalten für die CSV
column_names = [
    'text',
    'stance',
    'sourceID',
    'previousArgumentInSourceId',
    'acquisitionTime',
    'discussionTitle',
    'sourceTitle',
    'sourceUrl',
    'nextArgumentInSourceId',
    'id',
    'conclusion',
]

with open(json_path) as f_in:
    data = json.load(f_in)
    arguments = data['arguments']

with open(csv_path, 'w', newline='', encoding='utf-8') as f_out:
    csv_writer = csv.writer(f_out, delimiter=',')
    csv_writer.writerow(column_names)

    for argument in tqdm(arguments):
        row = []
        row.extend([value for value in argument['premises'][0].values()])
        row.extend([value for value in argument['context'].values()])
        row.append(argument['id'])
        row.append(argument['conclusion'])

        csv_writer.writerow(row)
