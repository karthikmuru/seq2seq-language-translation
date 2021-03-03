import tensorflow as tf
import json
from util import Util

class VocabEntry():
  def __init__(self, path=None, num_examples=None):
    if path == None: return None

    _, tokenizer = Util.load_dataset(path, num_examples)
    
    self.tokenizer = tokenizer
    self.word2index = tokenizer.word_index
    self.index2word = tokenizer.index_word
  
  def size(self):
    return len(self.word2index)

class Vocab():
  def __init__(self, src, tgt):
    self.src = src
    self.tgt = tgt

  def save(self, path):
    data = {
        'src_word2index': self.src.word2index,
        'src_index2word': self.src.index2word,
        'tgt_word2index': self.tgt.word2index,
        'tgt_index2word': self.tgt.index2word
    }
    json.dump(data, open(path, 'w'), indent=2)
 
  @staticmethod
  def load(self, path):
    src = VocabEntry()
    tgt = VocabEntry()

    data = json.load(open(path, 'r'))

    src.word2index = data['word2index']
    src.index2word = data['index2word']

    tgt.word2index = data['word2index']
    tgt.index2word = data['index2word']

    return Vocab(src, tgt)

  @staticmethod
  def build(src_path, tgt_path, num_examples=None):

    src = VocabEntry(src_path, num_examples)
    tgt = VocabEntry(tgt_path, num_examples)

    return Vocab(src, tgt)