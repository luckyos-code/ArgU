from tqdm import tqdm

from argU import settings
from argU.database.mongodb import MongoDB
from argU.indexing.models import CBOW
from argU.utils.reader import get_queries


def create_trec_files():
    create_arguments_trec_file()
    create_queries_trec_file()


def create_arguments_trec_file():
    mongo_db = MongoDB()
    with open(settings.TREC_PATH, 'w', encoding='utf8') as f_out:
        for arg in tqdm(mongo_db.args_coll.find()):
            f_out.write(
                '<DOC>\n'
                f'<DOCNO>{arg["_id"]}</DOCNO>\n'
                f'{arg["premises"][0]["model_text"]}\n'
                '</DOC>\n'
            )


def create_queries_trec_file():
    queries = get_queries(CBOW.load())

    with open(settings.TREC_PATH_TOPICS, 'w', encoding='utf8') as f_out:
        for query in queries:
            f_out.write(
                '<top>\n'
                f'<num>{query.id}</num>\n'
                '<title>\n'
                f'{query.text}\n'
                '</title>\n'
                '</top>\n'
            )
