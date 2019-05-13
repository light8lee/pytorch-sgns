# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse
import six
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='vocab.txt', help="corpus path for building vocab")
    parser.add_argument('--corpus', type=str, default='./data/corpus.txt', help="corpus path")
    parser.add_argument('--unk', type=str, default='[UNK]', help="UNK token")
    parser.add_argument('--window', type=int, default=5, help="window size")
    return parser.parse_args()



def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class Preprocess(object):

    def __init__(self, window=5, unk='[UNK]', data_dir='./data/'):
        self.window = window
        self.unk = unk
        self.data_dir = data_dir

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]
        return iword, [self.unk for _ in range(self.window - len(left))] + left + right + [self.unk for _ in range(self.window - len(right))]

    def build(self, filepath):
        print("building vocab...")
        # step = 0
        # self.wc = {self.unk: 1}
        # with codecs.open(filepath, 'r', encoding='utf-8') as file:
        #     for line in file:
        #         step += 1
        #         if not step % 1000:
        #             print("working on {}kth line".format(step // 1000), end='\r')
        #         line = line.strip()
        #         if not line:
        #             continue
        #         sent = line.split()
        #         for word in sent:
        #             self.wc[word] = self.wc.get(word, 0) + 1
        # print("")
        # self.idx2word = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        # self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        # self.vocab = set([word for word in self.word2idx])
        self.idx2word = []
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = convert_to_unicode(line)
                if not word:
                    break
                word = word.strip()
                self.idx2word.append(word)
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.vocab = set(self.idx2word)
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        print("build done")

    def convert(self, filepath):
        print("converting corpus...")
        step = 0
        data = []
        self.wc = defaultdict(int)
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                sent = []
                for word in line.split():
                    if word in self.vocab:
                        sent.append(word)
                        self.wc[word] += 1
                    else:
                        sent.append(self.unk)
                        self.wc[self.unk] += 1
                for i in range(len(sent)):
                    iword, owords = self.skipgram(sent, i)
                    data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))
        print("")
        pickle.dump(data, open(os.path.join(self.data_dir, 'train.dat'), 'wb'))
        pickle.dump(self.wc, open(os.path.join(self.data_dir, 'wc.dat'), 'wb'))
        print("conversion done")


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(window=args.window, unk=args.unk, data_dir=args.data_dir)
    preprocess.build(args.vocab)
    preprocess.convert(args.corpus)
