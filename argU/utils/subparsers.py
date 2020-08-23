import json

import numpy as np

from argU import settings
from argU.database.mongodb import MongoDB
from argU.indexing.models import CBOW, InEmbedding, OutEmbedding, Desm
from argU.preprocessing.trec import create_trec_files
from argU.utils.reader import get_queries


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
        self.parser.add_argument('-i', '--init', action='store_true',
                                 help='Reinitialize MongoDB with preprocessed arguments')
        self.parser.add_argument('-n', '--number', type=int, default=-1,
                                 help='Number of arguments to read into MongoDB')
        self.parser.add_argument('-e', '--embeddings', action='store_true',
                                 help='Add IN and OUT embeddings to all arguments')
        self.parser.add_argument('-s', '--sentiments', action='store_true',
                                 help='Add sentiment values to all arguments')
        self.parser.add_argument('-d', '--desm', action='store_true',
                                 help='Create DESM collections')
        self.parser.add_argument('--arg_emb', choices=['in_emb', 'out_emb'], default='in_emb',
                                 help='Choose the arguments embedding type for DESM')

    def _run(self, args):
        if args.init:
            MongoDB(overwrite=True, max_args=args.number)
        elif args.embeddings:
            cbow = CBOW.load()
            MongoDB().add_args_embeddings(
                in_emb_model=InEmbedding(cbow=cbow),
                out_emb_model=OutEmbedding(cbow=cbow),
            )
        elif args.sentiments:
            MongoDB().add_args_sentiments()
        elif args.desm:
            desm = Desm(emb_type=args.arg_emb)
            MongoDB().create_desm_collection(desm=desm)
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
        print(f'"{arg["context"]["discussionTitle"]}" -> {most_sim}')

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

    def init_args(self, subparsers):
        self.parser = subparsers.add_parser(self.name, help='Run the DESM model and generate scores')
        self.parser.add_argument('-e', '--embedding', choices=['in_emb', 'out_emb'], default='in_emb',
                                 help='Choose the embedding type')

    def _run(self, args):
        desm = Desm(emb_type=args.embedding)
        desm.print_examples(queries_num=4, args_topn=5)


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
