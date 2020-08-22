from argU.management import read_command_line, run_commands

args_parsed = read_command_line()
run_commands(args_parsed)

# if argparsed.desm:
#     desm = DESM(CBOW.load())
#     top_args = desm.evaluate_queries(
#         desm.queries_to_emb(queries),
#         coll_emb,
#         top_n=4000,
#         max_args=-1,
#     )
#     desm.store_query_results(coll_res, queries, top_args)
#
# if argparsed.merge:
#
#     N = 1000
#     print(f'N Value: {N}')
#
#     max_queries = -1
#
#     output_dict = dict()
#     for i, desm_scores in enumerate(coll_res.find()):
#         if i == max_queries:
#             break
#
#         query_id = desm_scores['query_id']
#         args = desm_scores['args'][:N]
#
#         terrier_data = dict()
#         with open(settings.TERRIER_RESULTS_PATH, 'r') as f_in:
#             for line in f_in:
#                 line = line.split()
#                 if line[0] == query_id:
#                     terrier_data[int(line[2])] = line[4]  # Arg
#         merged_args = []
#         for a in args:
#             if a in terrier_data:
#                 sents = coll_sents.find_one({'_id': a})
#                 if sents is None:
#                     print("BAD================================")
#                     sents = {'score': -0.1}
#                 merged_args.append(
#                     (a, float(terrier_data[a]), sents['score'])
#                 )
#
#         merged_args.sort(key=lambda x: x[1], reverse=True)
#         # merged_args = merged_args[:20]
#
#         if argparsed.sentiments != 'no':
#             merged_args_with_sents = []
#             for ma in merged_args:
#                 dph, sent = ma[1], ma[2]
#                 if argparsed.sentiments == 'emotional':
#                     dph = dph + dph * (abs(sent) / 2)
#                 elif argparsed.sentiments == 'neutral':
#                     dph = dph - dph * (abs(sent) / 2)
#                 merged_args_with_sents.append(
#                     (ma[0], dph, sent)
#                 )
#             merged_args = merged_args_with_sents
#             merged_args.sort(key=lambda x: x[1], reverse=True)
#
#         print(f'### {query_id} {desm_scores["query_text"]}')
#         # print('---')
#         # arguments.fancy_print(
#         #     coll_args,
#         #     merged_args_list[:20],
#         #     trans_dict=trans_dict,
#         #     arg_len=2000,
#         # )
#
#         # Sentiment Analysis
#
#         if len(merged_args) != 0:
#             output_dict[query_id] = merged_args
#         else:
#             output_dict[query_id] = [(
#                 '10113b57-2019-04-18T17:05:08Z-00001-000',
#                 0.0,
#                 0.0,
#             )]
#
#     with open(os.path.join(argparsed.output, 'run.txt'), 'w') as f_out:
#         method = settings.METHOD_NO
#         if argparsed.sentiments == 'emotional':
#             method = settings.METHOD_EMOTIONAL
#         elif argparsed.sentiments == 'neutral':
#             method = settings.METHOD_NEUTRAL
#         for (id, args) in output_dict.items():
#             for i, (arg_id, score, sent) in enumerate(args):
#                 trans_id = ''
#                 try:
#                     trans_id = coll_trans.find_one({'_id': arg_id})['arg_id']
#                 except:
#                     trans_id = arg_id
#
#                 f_out.write(' '.join([
#                     str(id), 'Q0', trans_id, str(i + 1), str(score), method, '\n'
#                 ]))
