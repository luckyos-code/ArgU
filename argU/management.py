import argparse

from argU.utils.subparsers import MongoDBSubparser, EmbeddingSubparser, Subparser, DESMSubparser, TrecSubparser, \
    MappingSubparser, EvalSubparser, DefaultSubparser

mongodb_subparser = MongoDBSubparser(name='mongodb', help='Store to / Read from MongoDB collections')
embedding_subparser = EmbeddingSubparser(name='embedding', help='CBOW model options')
desm_subparser = DESMSubparser(name='desm', help='Run the DESM model and generate scores')
trec_subparser = TrecSubparser(name='trec', help='Generate Trec files for Terrier')
mapping_subparser = MappingSubparser(name='mapping', help='Generate a Mapping for shorter argument IDs')
evaluation_subparser = EvalSubparser(name='eval', help='Evaluate Terrier and DESM results')


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
