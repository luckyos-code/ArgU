import re

import nltk

from argU import settings

with open(settings.STOPWORDS_PATH, "r") as f_in:
    stopwords = f_in.read().split("\n")


class Pipeline:
    def __init__(self, next_pipeline=None):
        self.next_pipeline = next_pipeline

    def __call__(self, text):
        text = self._run_token_pipeline(text)
        text = self._run_text_pipeline(text)
        return self._run_next_pipeline(text)

    def non_critical_token(self, token):
        return token.isalpha() or (token[:-1].isalpha() and token[-1] in [',', '.', '?', '!', ':', ';'])

    def _run_next_pipeline(self, text):
        if self.next_pipeline is not None:
            return self.next_pipeline(text)
        return text

    def _run_token_pipeline(self, text):
        tokens = []
        for token in text.split():
            new_token = self._process_token(token)
            tokens.extend(new_token.split())
        return ' '.join(tokens)

    def _run_text_pipeline(self, text):
        raise NotImplemented

    def _process_token(self, token):
        raise NotImplemented


class PreprocessorPipeline(Pipeline):

    def __init__(self, next_pipeline=None):
        super().__init__(next_pipeline)

    def _process_token(self, token):
        token = StringTools.replace_same_letter_substrings(token)

        if self.non_critical_token(token):
            token = StringTools.replace_upper_tokens(token)
            return token

        token = StringTools.remove_square_brackets(token)
        token = StringTools.remove_duplicate_sentence_marks(token)
        token = StringTools.replace_special_non_ascii_chars(token)

        token = StringTools.stretch_three_dots(token)
        token = StringTools.stretch_round_brackets(token)
        token = StringTools.stretch_commas(token)
        token = StringTools.stretch_by_special_chars_before_url_cleaning(token)

        token = StringTools.replace_urls_by_regex(token)
        token = StringTools.replace_urls_by_domain(token)

        token = StringTools.stretch_by_special_chars_after_url_cleaning(token)
        token = StringTools.remove_too_long_tokens(token)
        return token

    def _run_text_pipeline(self, text):
        text = StringTools.fix_round_brackets(text)
        text = StringTools.fix_commas(text)
        return text


class ModelPostPipeline(Pipeline):

    def __init__(self, next_pipeline=None):
        super().__init__(next_pipeline)

    def _process_token(self, token):
        token = StringTools.remove_round_brackets_in_one_token(token)
        token = self._replace_special_chars(token)
        token = self._tokenize_number(token)
        token = self._remove_stopwords(token)
        return token

    def _run_text_pipeline(self, text):
        return text

    def _replace_special_chars(self, token):
        token = re.sub(r'[,".!?]', '', token)
        token = re.sub(r' - |:|;', ' ', token)
        return token

    def _tokenize_number(self, token):
        token = re.sub(r'\d+', StringTools.NUM_TOKEN, token)
        return token

    def _remove_stopwords(self, token):
        if token.lower() in stopwords:
            return ''
        return token


class ApiPostPipeline(Pipeline):

    def __init__(self, next_pipeline=None):
        super().__init__(next_pipeline)

    def _process_token(self, token):
        token = StringTools.remove_round_brackets_in_one_token(token)
        token = StringTools.remove_special_token(token)
        token = StringTools.correct_apostrophes(token)
        return token

    def _run_text_pipeline(self, text):
        return text


class QueryPipeline(Pipeline):

    def __init__(self, cbow, next_pipeline=None):
        self.emb_model = cbow.model
        super().__init__(next_pipeline)

    def _process_token(self, token):
        token = self._match_pos_tag_and_case(token)
        if not self._token_in_embeddings(token):
            token = self._find_better_token(token)

        return token

    def _match_pos_tag_and_case(self, token):
        if not self._is_noun(token):
            return token.lower()
        return token

    def _token_in_embeddings(self, token):
        return token in self.emb_model.wv

    def _find_better_token(self, token):
        if self._token_in_embeddings(token.lower()):
            return token.lower()
        else:
            return self._clear_hyphens(token)

    def _clear_hyphens(self, token):
        subtokens = []
        for subtoken in token.split('-'):
            if self._token_in_embeddings(subtoken):
                subtokens.append(subtoken)
            elif self._token_in_embeddings(subtoken.lower()):
                subtokens.append(subtoken.lower())
            else:
                return token

        return ' '.join(subtokens)

    def _is_noun(self, token):
        return 'NN' in self._pos_tag(token)

    def _pos_tag(self, token):
        try:
            return nltk.pos_tag([token])[0][1]
        except LookupError as le:
            nltk.download('averaged_perceptron_tagger')
            return nltk.pos_tag([token])[0][1]

    def _run_text_pipeline(self, text):
        return text


model_nlp_pipeline = PreprocessorPipeline(ModelPostPipeline())
api_nlp_pipeline = PreprocessorPipeline(ApiPostPipeline())


class StringTools:
    URL_TOKEN = '<URL>'
    NUM_TOKEN = '<NUM>'
    special_tokens = [URL_TOKEN, NUM_TOKEN]

    @staticmethod
    def remove_square_brackets(token):
        return re.sub(r'\[(.*?)\]', ' ', token)

    @staticmethod
    def remove_too_long_tokens(token):
        if len(token) >= 45:
            return ''
        return token

    @staticmethod
    def remove_round_brackets_in_one_token(token):
        if token and token[0] == '(' and token[-1] == ')':
            return ''
        return token

    @staticmethod
    def replace_special_non_ascii_chars(token):
        token = re.sub(r'[“”]', '"', token)
        token = re.sub(r'[‘’]', '\'', token)
        token = re.sub(r'[ם…∼˜■●•ï¿½¾ⓒ†لرَّحِيم€™∴©ℵ∞ಠ_ಠٱه—–·­±≠⋅בְּרֵשִׁיתהומוסקסואלœ0ָָ]', '', token)
        token = re.sub(r'[‚]', ',', token)
        return re.sub(r'[⇒≡≈]', '=', token)

    @staticmethod
    def stretch_three_dots(token):
        return re.sub(r'\.{3,}', " ... ", token)

    @staticmethod
    def stretch_round_brackets(token):
        token = token.replace(')', ' ) ')
        return token.replace('(', ' ( ')

    @staticmethod
    def stretch_by_special_chars_before_url_cleaning(token):
        return re.sub(r'[<>]', ' ', token)

    @staticmethod
    def stretch_commas(token):
        return token.replace(',', ' , ')

    @staticmethod
    def replace_urls_by_regex(token):
        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_regex, f' {StringTools.URL_TOKEN} ', token)

    @staticmethod
    def replace_urls_by_domain(token):
        if any(end in token for end in ('.com', '.net', '.org', '.edu')):
            return ''
        return token

    @staticmethod
    def stretch_by_special_chars_after_url_cleaning(token):
        return re.sub(r'[;:/~#§&@=]', ' ', token)

    @staticmethod
    def remove_duplicate_sentence_marks(token):
        token = re.sub(r'\?{2,}', '?', token)
        token = re.sub(r'!{2,}', '!', token)
        token = re.sub(r'\*{2,}', '', token)
        return re.sub(r'(\?!|!\?)+', '?!', token)

    @staticmethod
    def replace_same_letter_substrings(token):
        return re.sub(r'([a-zA-Z])\1{3,}', r'\1', token)

    @staticmethod
    def replace_upper_tokens(token):
        if len(token) >= 5 and token.isupper():
            return token.lower()
        return token

    @staticmethod
    def clean_sub_points(text):
        text = re.sub(r'([1-9]+[0-9]*[-][A-Za-z])', r' \1 ', text)
        text = re.sub(r'([1-9]+[0-9]*[).])', r' \1 ', text)
        return text

    @staticmethod
    def fix_round_brackets(text):
        text = text.replace(' )', ')')
        return text.replace('( ', '(')

    @staticmethod
    def fix_commas(text):
        return text.replace(' ,', ',')

    @staticmethod
    def remove_special_token(token):
        if token in StringTools.special_tokens:
            return ''
        return token

    @staticmethod
    def correct_apostrophes(token):
        if '"' in token[1:-1]:
            matches = re.findall(r'[a-z]\"[a-z]', token)
            if matches and len(token.split('"')[1]) <= 2:
                return re.sub(r'([a-z])\"([a-z])', r"\1'\2", token)
        return token

    @staticmethod
    def lower_first_letter(token):
        return token[0].lower() + token[1:]


def token_variants(token):
    seq = [token, StringTools.lower_first_letter(token), token.lower()]
    seen = set()

    return [x for x in seq if not (x in seen or seen.add(x))]
