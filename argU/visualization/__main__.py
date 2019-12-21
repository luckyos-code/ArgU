from indexing.models import Argument2Vec, CBOW, Text2Vec
from utils.reader import DebateIterator, ArgumentIterator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
import rootpath
import sys
import numpy as np
import pandas as pd

ROOT_PATH = rootpath.detect()
RESOURCES_PATH = os.path.join(ROOT_PATH, 'resources/')
IMAGES_PATH = os.path.join(RESOURCES_PATH, 'images/')
CSV_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')

MODEL_TYPE = 'debate'
MODEL_PATH = os.path.join(RESOURCES_PATH, f'cbow.{MODEL_TYPE}.model')

pca = PCA(n_components=2)

cbow = CBOW()
cbow.load(MODEL_PATH)

query = 'gay marriage'
max_args = 2000
max_debates = 2000

if MODEL_TYPE == 'args':
    arguments = ArgumentIterator(CSV_PATH, max_args=max_args)
    t2v = Argument2Vec(cbow.model, arguments)
elif MODEL_TYPE == 'debate':
    debates = DebateIterator(CSV_PATH, max_debates=max_debates)
    t2v = Text2Vec(cbow.model, debates)

arguments, similarities = t2v.most_similar(query, topn=-1)

argument_matrix = []
for arg_id in arguments:
    argument_matrix.append(t2v.tv[arg_id])
argument_matrix = np.asarray(argument_matrix)
principal_components = pca.fit_transform(argument_matrix)
principal_df = pd.DataFrame(
    data=principal_components,
    columns=['pc_1', 'pc_2'],
)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC_1', fontsize=10)
ax.set_ylabel('PC_2', fontsize=10)
ax.set_title(f"Query: {query}", fontsize=14)

ax.scatter(
    principal_df['pc_1'],
    principal_df['pc_2'],
    s=4,
    c=similarities,
    cmap='seismic'
)

ax.grid()
# plt.show()

image_path = os.path.join(IMAGES_PATH, f"{query}_{max_args}_{MODEL_TYPE}.png")
plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
