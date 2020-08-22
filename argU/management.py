import argparse

from argU.utils.subparsers import MongoDBSubparser, EmbeddingSubparser, Subparser, DESMSubparser, TrecSubparser

mongodb_subparser = MongoDBSubparser('mongodb')
embedding_subparser = EmbeddingSubparser('embedding')
desm_subparser = DESMSubparser('desm')
trec_subparser = TrecSubparser('trec')


def read_command_line():
    parser = argparse.ArgumentParser(prog='ArgU')
    subparsers = parser.add_subparsers(help='ArgU Commands:', dest='subparser')

    parser_eval = subparsers.add_parser('eval', help='Generate TIRA results')

    mongodb_subparser.init_args(subparsers)
    embedding_subparser.init_args(subparsers)
    desm_subparser.init_args(subparsers)
    trec_subparser.init_args(subparsers)

    return parser.parse_args()


def run_commands(args):
    subparser_name = vars(args)['subparser']
    subparser = Subparser.manager.get(subparser_name, None)
    subparser(args)
