import json

from argU import settings
from argU.utils.reader import get_mapped_ids_to_sentiments, get_mapping_to_arg_id


class Argument:
    def __init__(self, *, id, cos):
        self.id = id
        self.cos = cos
        self.dph = None
        self.sent = None
        self.final_score = None

    def __repr__(self):
        return f'{self.__class__.__name__}(id: {self.id}, cos: {self.cos}, dph: {self.dph}, sent: {self.sent}, final: {self.final_score})'

    def __eq__(self, other):
        if isinstance(other, Argument):
            return self.final_score == other.final_score
        return False

    def __lt__(self, other):
        return self.final_score < other.final_score


class ResultManager:
    methods = {
        'emotional': settings.METHOD_EMOTIONAL,
        'neutral': settings.METHOD_NEUTRAL,
        'none': settings.METHOD_NO,
    }

    mapping = get_mapping_to_arg_id()

    def __init__(self, *, sent_type, emb_type, args_topn, store_path):
        self.emb_type = emb_type
        self.args_topn = args_topn
        self.sent_type = sent_type
        self.store_path = store_path
        self.method = self.methods[sent_type]
        self.terrier_result = TerrierResultManager(args_topn=args_topn)
        self.desm_result = DesmResultManager(emb_type=self.emb_type, args_topn=args_topn)
        self.scoring_functions = self._init_scoring_functions()

    def generate_results(self):
        merged_result = self._merge()
        self._add_sentiments(merged_result)
        self._add_final_scores(merged_result)
        self._sort_results(merged_result)
        self._generate_file(merged_result)

    def _init_scoring_functions(self):
        return {
            'emotional': ResultManager._final_score_sentiments_emotional,
            'neutral': ResultManager._final_score_sentiments_neutral,
            'none': ResultManager._final_score_sentiments_no,
        }

    def _sort_results(self, result):
        for query_id, args in result.items():
            result[query_id] = sorted(result[query_id], reverse=True)

    def _add_sentiments(self, result):
        sentiment_mapping = get_mapped_ids_to_sentiments()

        for query_id, args in result.items():
            for arg in args:
                arg.sent = sentiment_mapping[arg.id]

    def _add_final_scores(self, result):
        for query_id, args in result.items():
            for arg in args:
                arg.final_score = self._final_score(arg)

    def _merge(self):
        merged_results = {}
        for query_id, desm_args in self.desm_result.result.items():
            merged_results[query_id] = self._merged_desm_terrier_args(query_id, desm_args)

        return merged_results

    def _merged_desm_terrier_args(self, query_id, desm_args):
        result = []
        for arg in desm_args:
            new_arg = self._terrier_merge(arg, self.terrier_result.result[query_id])
            if new_arg is not None:
                result.append(new_arg)

        return result

    def _terrier_merge(self, arg, terrier_result):
        terrier_dph = terrier_result.get(arg.id, None)
        if terrier_dph is not None:
            arg.dph = terrier_dph
            return arg

    def _final_score(self, arg):
        return self.scoring_functions[self.sent_type](arg.dph, arg.sent)

    def _generate_file(self, result):
        with open(self.store_path, 'w') as f_out:
            for query_id, args in result.items():
                self._write_args_to_file(query_id, args, f_out)

    def _write_args_to_file(self, query_id, args, file):
        if len(args) == 0:
            self._write(query_id, '10113b57-2019-04-18T17:05:08Z-00001-000', 0, 0.0, file)
        else:
            for i, arg in enumerate(args):
                original_arg_id = self.mapping[arg.id]
                self._write(query_id, original_arg_id, i, arg.final_score, file)

    def _write(self, query_id, arg_id, pos, score, file):
        file.write(' '.join([
            str(query_id),
            'Q0',
            arg_id,
            str(pos + 1),
            str(score),
            self.method,
            '\n'
        ]))

    @staticmethod
    def _final_score_sentiments_no(dph, sent):
        return dph

    @staticmethod
    def _final_score_sentiments_emotional(dph, sent):
        return dph + dph * (abs(sent) / 2)

    @staticmethod
    def _final_score_sentiments_neutral(dph, sent):
        return dph - dph * (abs(sent) / 2)


class DesmResultManager:
    def __init__(self, emb_type, args_topn):
        self.args_topn = args_topn
        self.emb_type = emb_type
        self.path = ''
        self.result = {}

        self._init()

    def _init(self):
        self.path = settings.get_desm_results_path(self.emb_type)
        self.result = self._get_result_dict()

    def _get_result_dict(self):
        data = self._read_data()
        result = {}

        for query_dict in data:
            query_id, args = self._query_data(query_dict)
            result[int(query_id)] = self._arg_object_list(args[:self.args_topn])

        return result

    def _arg_object_list(self, args):
        result = []

        for arg in args:
            result.append(Argument(id=arg['id'], cos=arg['cos']))

        return result

    def _read_data(self):
        with open(self.path, 'r') as f_in:
            return json.load(f_in)

    def _query_data(self, query_dict):
        return tuple(query_dict.items())[0]


class TerrierResultManager:
    def __init__(self, args_topn):
        self.args_topn = args_topn
        self.result = {}
        self._init()

    def _init(self):
        self.result = self._get_result_dict()

    def _get_result_dict(self):
        result = {}

        with open(settings.TERRIER_RESULTS_PATH, 'r') as f_in:
            for row in f_in:
                self._add_row_to_result(row, result)

        return result

    def _add_row_to_result(self, row, result):
        query_id, arg_id, score, = self._process_terrier_row(row)
        if self._args_limit_not_exceeded(result.get(query_id, {})):
            self._add_row(arg_id, score, query_id, result)

    def _add_row(self, arg_id, dph_score, query_id, result):
        if query_id not in result:
            result[query_id] = {arg_id: dph_score}
        else:
            result[query_id][arg_id] = dph_score

    def _args_limit_not_exceeded(self, args):
        return len(args) < self.args_topn

    def _process_terrier_row(self, row):
        query_id, _, arg_id, _, score, _ = row.split()
        query_id = int(query_id)
        arg_id = int(arg_id)
        score = float(score)

        return query_id, arg_id, score
