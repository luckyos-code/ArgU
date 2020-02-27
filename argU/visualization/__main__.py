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

pca = PCA(n_components=2)
cbow = CBOW.load()

db = load_db()
coll_emb = db[setup.MONGO_DB_COL_EMBEDDINGS]
coll_emb_back = db[setup.MONGO_DB_COL_EMBEDDINGS_BACKUP]

args_emb = []
args_emb_back = []
max_args = 2000

for i, arg in tqdm(enumerate(coll_emb.find())):
    if i == max_args:
        break
    args_emb.append(arg['emb'])

for i, arg in tqdm(enumerate(coll_emb_back.find())):
    if i == max_args:
        break
    args_emb_back.append(arg['emb'])

args_emb = np.asarray(args_emb)
args_emb_back = np.asarray(args_emb_back)

print(f'Arg Emb shape: {args_emb.shape}')
print(f'Arg Emb Backup shape: {args_emb_back.shape}')

all_args = np.concatenate((args_emb, args_emb_back))
print(f'All shape: {all_args.shape}')

principal_components = pca.fit_transform(all_args)
principal_df = pd.DataFrame(
    data=principal_components,
    columns=['pc_1', 'pc_2'],
)

print(principal_df.shape)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC_1', fontsize=10)
ax.set_ylabel('PC_2', fontsize=10)

ax.scatter(
    principal_df['pc_1'][:2000],
    principal_df['pc_2'][:2000],
    s=4,
    c='red',
)

ax.scatter(
    principal_df['pc_1'][2000:],
    principal_df['pc_2'][2000:],
    s=4,
    c='blue',
)

ax.grid()
plt.show()

# image_path = os.path.join(IMAGES_PATH, f"{query}_{max_args}_{MODEL_TYPE}.png")
# plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
