import json
# from sklearn.ensemble import ExtraTreesClassifier
import os

import nltk
from WikiLoaderv2 import WikiDataLoader
from pprint import pprint
from nltk import RegexpTokenizer
from copy import copy
import pickle
import sys
import lightgbm as lgb
import numpy as np
from tqdm import tqdm

# https://www.kaggle.com/gabrielaltay/categorical-variables-in-decision-trees


class Tokenizer:
    def __init__(self):
        self.tokenize = self.create_subword_tokenizer("en", vs=100000)
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    @staticmethod
    def create_subword_tokenizer(lang, vs):
        from pathlib import Path
        from bpemb.util import sentencepiece_load, http_get
        import re

        def _load_file(file, archive=False):
            cache_dir = Path.home() / Path(".cache/bpemb")
            archive_suffix = ".tar.gz"
            base_url = "https://nlp.h-its.org/bpemb/"
            cached_file = Path(cache_dir) / file
            if cached_file.exists():
                return cached_file
            suffix = archive_suffix if archive else ""
            file_url = base_url + file + suffix
            print("downloading", file_url)
            return http_get(file_url, cached_file, ignore_tardir=True)

        model_file = "{lang}/{lang}.wiki.bpe.vs{vs}.model".format(lang=lang, vs=vs)
        model_file = _load_file(model_file)
        spm = sentencepiece_load(model_file)
        return lambda text: spm.EncodeAsPieces(re.sub(r"\d", "0", text.lower()))


class Batcher:
    def __init__(self, corpus, max_len=25, max_batch_len=10000):
        self.max_len = max_len
        self.max_batch_len = max_batch_len
        self.tokenizer = Tokenizer()
        self.word_dict = {"-": 0}
        self.corpus = corpus

    def prepare_sentence(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)

        batch = []

        for token_ind, token in enumerate(tokens):

            empty = [0] * self.max_len

            if token not in self.word_dict:
                self.word_dict[token] = len(self.word_dict)

            for ind, previous in enumerate(range(token_ind - 1, -1, -1)):
                if token_ind - ind >= self.max_len:
                    break

                empty[previous + self.max_len - token_ind] = self.word_dict[tokens[previous]]

            batch.append((copy(empty), self.word_dict[token]))

        return batch

    @property
    def num_classes(self):
        return len(self.word_dict)

    def __iter__(self):
        batch = []
        for doc in self.corpus:
            doc = json.loads(doc)['text']
            for sent in self.tokenizer.sent_tokenizer.tokenize(doc):
                batch_data = self.prepare_sentence(sent)

                batch.extend(batch_data)

                if len(batch) >= self.max_batch_len:
                    X, y = zip(*batch)
                    yield X, y
                    batch = []
        if len(batch) > 0:
            X, y = zip(*batch)
            yield X, y

    def save_worddict(self):
        pickle.dump(self.word_dict, open("word_dict.pkl", "wb"))


class LightGBMDecisionTree:
    
    def __init__(self, num_class):
        self.params = {
            'objective': 'multiclass',
            # 'bagging_freq': 0,
            # 'categorical_feature': list(range(MAX_LEN)),
            # 'boost_from_average': True,
            # 'num_leaves': 5,
            'num_class': num_class
        }
        
    def train(self, lgb_train):
        self.gbm = lgb.train(
            self.params, 
            lgb_train, 
            num_boost_round=1,
            categorical_feature='auto')
    
    def predict(self, X):
        return self.gbm.predict(X)


def main(args):
    wiki = WikiDataLoader(args.wiki_path)
    batcher = Batcher(wiki)

    total_processed = 0
    tree_count = 0

    if not os.path.isdir("trees"):
        os.mkdir("trees")

    try:
        for X, y in tqdm(batcher):
            data = lgb.Dataset(np.array(X), label=np.array(y))

            tree = LightGBMDecisionTree(num_class=batcher.num_classes)
            tree.train(data)

            tree_count += 1
            total_processed += len(X)
    except KeyboardInterrupt:
        pass

    batcher.save_worddict()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("wiki_path")

    args = parser.parse_args()

    main(args)