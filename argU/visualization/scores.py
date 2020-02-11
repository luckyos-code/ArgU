import matplotlib.pyplot as plt
import csv
import json
import os
import sys
import rootpath

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
except Exception as e:
    print("Project intern dependencies could not be loaded...")
    print(e)
    sys.exit(0)


def draw_query_scores(alpha=None):

    with open(setup.SCORES_PATH, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.reader(f_in, **setup.SCORES_CONFIG)

        for (_, query_text, top_args) in reader:
            top_args = json.loads(top_args)
            ordered_tuples = [(id, *scores) for id, scores in top_args.items()]
            ordered_tuples.sort(key=lambda x: x[1], reverse=True)

            final = []
            desim = []
            bm25 = []
            sents = []

            for (_, fs, bs, ds, sent, sent_m) in ordered_tuples:
                if alpha is None:
                    final.append(fs)
                else:
                    final.append(alpha * bs + (1 - alpha) * ds)
                bm25.append(bs)
                desim.append(ds)
                sents.append(sent)

            file_name = '_'.join(query_text.split()) + '.png'
            file_path = os.path.join(setup.IMAGES_PATH, file_name)

            x = list(range(0, len(desim)))
            plt.title(file_name)
            desim_dots, = plt.plot(x, desim, 'ro', ms=2.5, label='CBOW')
            bm25_dots, = plt.plot(x, bm25, 'go', ms=2.5, label='BM25')
            final_dots, = plt.plot(x, final, 'bo', ms=2.5, label='Combined')
            plt.legend(handles=[desim_dots, bm25_dots, final_dots])
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    draw_query_scores(alpha=None)
