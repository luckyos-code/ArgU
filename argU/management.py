import argparse

from argU.utils.subparsers import MongoDBSubparser, EmbeddingSubparser, Subparser, DESMSubparser, TrecSubparser, \
    MappingSubparser, EvalSubparser, DefaultSubparser

mongodb_subparser = MongoDBSubparser(name='mongodb')
embedding_subparser = EmbeddingSubparser(name='embedding')
desm_subparser = DESMSubparser(name='desm')
trec_subparser = TrecSubparser(name='trec')
mapping_subparser = MappingSubparser(name='mapping')
evaluation_subparser = EvalSubparser(name='eval')


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
    subparser = Subparser.manager.get(subparser_name, DefaultSubparser())
    subparser(args)
