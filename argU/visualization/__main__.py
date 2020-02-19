from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
import rootpath
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm


try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.indexing.models import CBOW
    from argU.indexing.models import DESM
    from argU.utils.arguments import TrainArgsIterator
    from argU.preprocessing.mongodb import load_db
except Exception as e:
    print(e)
    sys.exit(0)


cbow = CBOW.load()

db = load_db()
coll = db[setup.MONGO_DB_COL_EMBEDDINGS]

pca = PCA(n_components=2)
arguments = []

for arg in tqdm(coll.find()):
    arguments.append(arg['emb'])
arguments = np.asarray(arguments)

principal_components = pca.fit_transform(arguments)
principal_df = pd.DataFrame(
    data=principal_components,
    columns=['pc_1', 'pc_2'],
)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC_1', fontsize=10)
ax.set_ylabel('PC_2', fontsize=10)

ax.scatter(
    principal_df['pc_1'],
    principal_df['pc_2'],
    s=4,
)

ax.grid()
plt.show()

# image_path = os.path.join(IMAGES_PATH, f"{query}_{max_args}_{MODEL_TYPE}.png")
# plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
