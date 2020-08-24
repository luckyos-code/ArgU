import json

import numpy as np

from argU import settings
from argU.database.mongodb import MongoDB
from argU.indexing.models import CBOW, InEmbedding, OutEmbedding, Desm
from argU.preprocessing.trec import create_trec_files
from argU.results.results import ResultManager
from argU.utils.reader import get_queries, get_arg_id_to_mapping


class Subparser:
    manager = {}

    def __init__(self, name):
        self.name = name
        self.parser = None
        self.manager[self.name] = self

    def __call__(self, args):
        self._run(args)

    def init_args(self, subparsers):
        raise NotImplemented

    def _run(self, args):
        raise NotImplemented


class MongoDBSubparser(Subparser):
    def __init__(self, name):
        super().__init__(name)

    def init_args(self, subparsers):
        self.parser = subparsers.add_parser(self.name, help='MongoDB options')
        self.parser.add_argument('--find', default=None, help='Find Argument by ID')
        self.parser.add_argument('-i', '--init', action='store_true',
                                 help='Reinitialize MongoDB with preprocessed arguments')
        self.parser.add_argument('--num', type=int, default=-1,
                                 help='Number of arguments to read into MongoDB')
        self.parser.add_argument('-e', '--embeddings', action='store_true',
                                 help='Add IN and OUT embeddings to all arguments')
        self.parser.add_argument('--arg_emb', choices=['in_emb', 'out_emb'], default='in_emb',
                                 help='Choose the arguments embedding type for `-d`')

    def _run(self, args):
        if args.init:
            MongoDB().reinitialize(max_args=args.num)
        elif args.embeddings:
            cbow = CBOW.load()
            MongoDB().add_args_embeddings(
                in_emb_model=InEmbedding(cbow=cbow),
                out_emb_model=OutEmbedding(cbow=cbow),
            )
        elif args.find is not None:
            arg = MongoDB().args_coll.find_one({'id': args.find})
            print(arg)
        else:
            print(MongoDB())


class EmbeddingSubparser(Subparser):
    def __init__(self, name):
        super().__init__(name)

    def init_args(self, subparsers):
        self.parser = subparsers.add_parser(self.name, help='CBOW model options')
        self.parser.add_argument('-t', '--train', action='store_true',
                                 help='Train a new continuous bag of words model')
        self.parser.add_argument('-q', '--queries', action='store_true',
                                 help='Print First 5 Query Embeddings and their 5 most similar words')
        self.parser.add_argument('-a', '--arguments', action='store_true',
                                 help='Print First 5 Argument Embeddings and their 5 most similar words')
        self.parser.add_argument('-o', '--out', action='store_true',
                                 help='Print an example with out-embedding. In combination with -q, use out-embeddings.')

    def _run(self, args):
        if args.train:
            mongo_db = MongoDB()
            cbow = CBOW()
            cbow.train(mongo_db.model_texts_iterator(), min_count=3, size=300, window=6)

        elif args.out and not (args.queries or args.arguments):
            cbow = CBOW.load()
            cbow.switch_to_out_embedding()
            print(cbow)

        elif args.queries:
            cbow = CBOW.load()
            model = cbow.out_model if args.out else cbow.model
            query_emb_model = InEmbedding(cbow=cbow)
            queries = get_queries(cbow)

            self._print_queries(queries=queries, word_model=model, query_emb_model=query_emb_model)

        elif args.arguments:
            cbow = CBOW.load()
            model = cbow.out_model if args.out else cbow.model
            mongo_db = MongoDB()

            self._print_arguments(mongo_db=mongo_db, word_model=model, args_out=args.out)
        else:
            print(CBOW.load())

    def _print_arguments(self, *, mongo_db, word_model, args_out):
        for arg in mongo_db.args_coll.find().limit(10):
            emb = arg['premises'][0]['out_emb'] if args_out else arg['premises'][0]['in_emb']
            most_sim = word_model.wv.most_similar(positive=[np.asarray(emb)], topn=5)
            self._print_argument(arg, most_sim)

    def _print_argument(self, arg, most_sim):
        print(f'"{arg["conclusion"]}" -> {most_sim}')

    def _print_queries(self, *, queries, word_model, query_emb_model):
        for query in queries[:10]:
            query_emb = query_emb_model.text_to_emb(query.text)
            most_sim = word_model.wv.most_similar(positive=[query_emb], topn=5)
            self._print_query(query=query, most_sim=most_sim)

    def _print_query(self, *, query, most_sim):
        print(f'{query.text} -> {most_sim}')


class DESMSubparser(Subparser):
    def __init__(self, name):
        super().__init__(name)
        self.emb_type = None
        self.args_topn = None
        self.desm = None

    def init_args(self, subparsers):
        self.parser = subparsers.add_parser(self.name, help='Run the DESM model and generate scores')
        self.parser.add_argument('-s', '--store', action='store_true',
                                 help='Store results in a matching file')
        self.parser.add_argument('--emb', choices=['in_emb', 'out_emb'], default='in_emb',
                                 help='Choose the embedding type')
        self.parser.add_argument('--topn', type=int, default=100,
                                 help='How many Arguments per Query should be stored?')

    def _run(self, args):
        self._init_attributes(args)

        if args.store:
            self._store_in_file()
        else:
            self.desm.print_examples(queries_num=4, args_topn=5)

    def _init_attributes(self, args):
        self.emb_type = args.emb
        self.args_topn = args.topn
        self.desm = Desm(emb_type=self.emb_type)

    def _store_in_file(self):
        json_data = self._create_data()
        self._dump_data(json_data)

    def _create_data(self):
        data = []
        mapping = get_arg_id_to_mapping()

        for query_id, top_args in self.desm.query_results_iterator(args_topn=self.args_topn):
            arg_list = self._create_arg_list(top_args, mapping)
            data.append({query_id: arg_list})

        return data

    def _create_arg_list(self, args, mapping):
        arg_list = []
        for arg in args:
            arg_list.append(self._create_arg(arg, mapping))
        return arg_list

    def _create_arg(self, arg, mapping):
        return {
            'id': mapping[arg['id']],
            'cos': round(arg['cos_sim'], 6)
        }

    def _dump_data(self, data):
        path = settings.get_desm_results_path(self.emb_type)
        with open(path, 'w') as f_out:
            json.dump(data, f_out, separators=(',', ':'))


class MappingSubparser(Subparser):
    def __init__(self, name):
        super().__init__(name)

    def init_args(self, subparsers):
        self.parser = subparsers.add_parser(self.name, help='Generate Mapping for ALL arguments to numbers')

    def _run(self, args):
        mappings = self._create_mappings(self._read_args_me())
        self._store_mappings(mappings)

    def _read_args_me(self):
        with open(settings.ARGS_ME_JSON_PATH, 'r', encoding='utf8') as f_in:
            data = json.load(f_in)
        return data['arguments']

    def _create_mappings(self, arguments):
        mappings = {}
        for i, arg in enumerate(arguments):
            mappings[i] = arg['id']
        return mappings

    def _store_mappings(self, mappings):
        with open(settings.MAPPING_PATH, 'w') as f_out:
            json.dump(mappings, f_out, separators=(',', ':'))


class TrecSubparser(Subparser):
    def __init__(self, name):
        super().__init__(name)

    def init_args(self, subparsers):
        self.parser = subparsers.add_parser(self.name, help='Generate Trec files for Terrier')

    def _run(self, args):
        create_trec_files()


class EvalSubparser(Subparser):
    def __init__(self, name):
        super().__init__(name)

    def init_args(self, subparsers):
        self.parser = subparsers.add_parser(self.name, help='Evaluate Terrier and DESM results')
        self.parser.add_argument('--emb', choices=['in_emb', 'out_emb'], default='in_emb',
                                 help='Choose the embedding type')
        self.parser.add_argument('--args_topn', type=int, default=1000,
                                 help='Hoy many arguments from Terrier and DESM respectively should be used?')
        self.parser.add_argument('--sent', choices=['none', 'emotional', 'neutral'], default='none',
                                 help='Choose Method for argument after-ranking')
        self.parser.add_argument('--out', default=settings.OUR_RESULTS_PATH,
                                 help='Path for the result')

    def _run(self, args):
        result_manager = ResultManager(emb_type=args.emb, sent_type=args.sent, store_path=args.out,
                                       args_topn=args.args_topn)
        result_manager.generate_results()
