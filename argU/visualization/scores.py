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


def draw_query_scores(alpha=None, use_final=True, use_desim=True, use_bm25=True, use_sents=False):

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
            final_with_sents = []

            for i, (_, fs, bs, ds, sent, sent_m) in enumerate(ordered_tuples):
                if alpha is not None:
                    fs = alpha * bs + (1 - alpha) * ds
                final.append(fs)
                bm25.append(bs)
                desim.append(ds)
                sents.append(sent)

                print(i, bs, ds)
                f_sent = fs + fs * abs(sent)
                final_with_sents.append(f_sent)

            file_name = '_'.join(query_text.split()) + '.png'
            file_path = os.path.join(setup.IMAGES_PATH, file_name)

            x = list(range(0, len(desim)))
            plt.title(file_name)
            plt.figure(figsize=(16, 8))

            if use_sents:
                sent_dots, = plt.plot(x, sents, 'rx', ms=3.5, label='Sent')
                final_dots, = plt.plot(
                    x, final, 'bo', ms=3.5, label='Combined')
                final_sents, = plt.plot(
                    x, final_with_sents, 'gx', ms=3.5, label='Comb & Sent')
                plt.legend(handles=[final_dots, sent_dots, final_sents])
            else:
                desim_dots, = plt.plot(x, desim, 'ro', ms=3.5, label='CBOW')
                bm25_dots, = plt.plot(x, bm25, 'go', ms=3.5, label='BM25')
                final_dots, = plt.plot(
                    x, final, 'bo', ms=3.5, label='Combined')
                plt.legend(handles=[desim_dots, bm25_dots, final_dots])

            plt.savefig(file_path, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    draw_query_scores(alpha=None, use_sents=True)
