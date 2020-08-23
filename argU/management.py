import argparse

from argU.utils.subparsers import MongoDBSubparser, EmbeddingSubparser, Subparser, DESMSubparser, TrecSubparser, \
    MappingSubparser, EvalSubparser

mongodb_subparser = MongoDBSubparser('mongodb')
embedding_subparser = EmbeddingSubparser('embedding')
desm_subparser = DESMSubparser('desm')
trec_subparser = TrecSubparser('trec')
mapping_subparser = MappingSubparser('mapping')
evaluation_subparser = EvalSubparser('eval')


def read_command_line():
    parser = argparse.ArgumentParser(prog='ArgU')
    subparsers = parser.add_subparsers(help='ArgU Commands:', dest='subparser')

    mongodb_subparser.init_args(subparsers)
    embedding_subparser.init_args(subparsers)
    desm_subparser.init_args(subparsers)
    trec_subparser.init_args(subparsers)
    mapping_subparser.init_args(subparsers)
    evaluation_subparser.init_args(subparsers)

    return parser.parse_args()


def run_commands(args):
    subparser_name = vars(args)['subparser']
    subparser = Subparser.manager.get(subparser_name, None)
    subparser(args)
