import os
import csv
import json


def collect_scores(path, query_ids, query_texts, top_args, sentiments):
    with open(path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(
            f_out,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )

        result_log_header = ['id', 'query', 'top_args']
        writer.writerow(result_log_header)

        for query_id, query_text, (arg_ids, arg_fs, _, _), sents in zip(
            query_ids, query_texts, top_args, sentiments
        ):
            args_scores = dict()
            for id, fs, s0, s1 in zip(arg_ids, arg_fs, sents[0], sents[1]):
                args_scores[id] = [fs, s0, s1]
            line = [query_id, query_text, json.dumps(args_scores)]
            writer.writerow(line)


def scores_evaluate(scores_path):
    with open(scores_path, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.reader(
            f_in,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )

        header = next(reader)

        queries_args = []
        for (query_id, query_text, top_args) in reader:

            top_args = json.loads(top_args)
            ordered_tuples = [(id, *scores) for id, scores in top_args.items()]
            ordered_tuples.sort(key=lambda x: x[1], reverse=True)

            sorted_args = []

            # Entferne Argumente, deren desim < 0 ist
            for (arg, desim_score, sent, sent_magn) in ordered_tuples:
                if desim_score <= 0.01:
                    continue

                final_score = desim_score + desim_score * abs(sent)
                sorted_args.append(
                    (arg, final_score, desim_score, sent, sent_magn)
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

    sys.path.append(os.path.join(rootpath.detect(), 'argU'))

    from reader import FindArgumentIterator

    RESOURCES_PATH = os.path.join(rootpath.detect(), 'resources/')
    CSV_ARGS_PATH = os.path.join(RESOURCES_PATH, 'args-me.csv')
    SCORES_PATH = os.path.join(RESOURCES_PATH, 'scores.csv')

    queries_args = scores_evaluate(SCORES_PATH)

    for (query_id, query_text, args) in queries_args:
        print(f"Query \"{query_text}\" hat noch {len(args)} Argumente\n")

        arg_ids = [arg[0] for arg in args]
        arg_texts = dict()
        for arg in FindArgumentIterator(CSV_ARGS_PATH, arg_ids):
            arg_texts[arg.id] = arg.text_raw

        for arg in args[:20]:
            print(arg)
            print(f"\t > {arg_texts.get(arg[0], 'NOT FOUND')[:150]}")
        print(f"{'='*50}\n")

    # for arg in FindArgumentIterator(CSV_ARGS_PATH, []):
    # print(arg.text_raw)
    # print()
