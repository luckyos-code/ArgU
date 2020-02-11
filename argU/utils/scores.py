import os
import csv
import sys
import rootpath
import json

try:
    sys.path.append(os.path.join(rootpath.detect()))
    import setup
    from argU.preprocessing.tools import machine_model_clean
    from argU.preprocessing.tools import sentiment_clean
except Exception as e:
    print("Project intern dependencies could not be loaded...")
    print(e)
    sys.exit(0)


def collect_scores(query_ids, query_texts, top_args, sentiments):
    with open(setup.SCORES_PATH, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, **setup.SCORES_CONFIG)
        for query_id, query_text, (arg_ids, arg_fs, arg_bs, arg_ds), sents in zip(
            query_ids, query_texts, top_args, sentiments
        ):
            args_scores = dict()
            for id, fs, bs, ds, s0, s1 in zip(
                arg_ids, arg_fs, arg_bs, arg_ds, sents[0], sents[1]
            ):
                args_scores[id] = [fs, bs, ds, s0, s1]
            line = [query_id, query_text, json.dumps(args_scores)]
            writer.writerow(line)


def evaluate(threshold=0.009):
    with open(setup.SCORES_PATH, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.reader(f_in, **setup.SCORES_CONFIG)

        queries_args = []
        for (query_id, query_text, top_args) in reader:
            top_args = json.loads(top_args)
            ordered_tuples = [(id, *scores) for id, scores in top_args.items()]
            ordered_tuples.sort(key=lambda x: x[1], reverse=True)

            sorted_args = []

            # Entferne Argumente, deren desim <= Threshold ist
            for (arg_id, fs, bs, ds, sent, sent_m) in ordered_tuples:
                if fs <= threshold:
                    continue

                # sent_score = fs + fs * abs(sent)
                sent_score = fs
                sorted_args.append(
                    (arg_id, sent_score, fs, sent, sent_m)
                )
                sorted_args.sort(key=lambda x: x[1], reverse=True)

            queries_args.append((
                query_id,
                query_text,
                sorted_args,
            ))

        return queries_args


if __name__ == '__main__':
    import os
    import rootpath
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--threshold',
        default=0.05,
        type=float,
    )
    args = parser.parse_args()

    from argU.utils.reader import FindArgumentIterator

    queries_args = evaluate()

    for (query_id, query_text, args) in queries_args:
        print(f"Query \"{query_text}\" hat noch {len(args)} Argumente\n")

        arg_ids = [arg[0] for arg in args]
        arg_ids = arg_ids[:20]
        print(arg_ids)

        arg_texts = dict()
        for arg_id, arg_text in FindArgumentIterator(arg_ids, raw_texts_only=True):
            arg_texts[arg_id] = arg_text

        for arg_id in arg_ids:
            print(arg_id)
            print(f"\t > {arg_texts.get(arg_id, 'NOT FOUND')[:150]}")
        print(f"{'='*50}\n")
