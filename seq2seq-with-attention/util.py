import numpy as np
import tensorflow as tf
import re
import os
import io
import time

class Util():
  
  @staticmethod
  def preprocess(sent):
    sent = sent.lower().strip()
    sent = re.sub(r"([?.!,¿])", r" \1 ", sent)
    sent = re.sub(r'[" "]+', " ", sent)
    sent = '<start> ' + sent + ' <end>'
    return sent

  @staticmethod
  def text_to_sequence(text, tokenizer, padding=True):
    tensor = tokenizer.texts_to_sequences(text)
    
    if padding:
      tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor

  @staticmethod
  def load_dataset(path, num_lines = None):
    lines = io.open(path).read().strip().split('\n')[:num_lines]
    word_pairs = [ Util.preprocess(l) for l in lines ]    
    tensor, tokenizer = Util.tokenize(word_pairs)

    return tensor, tokenizer

  @staticmethod
  def tokenize(text):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = '')
    tokenizer.fit_on_texts(text)
    
    tensor = Util.text_to_sequence(text, tokenizer)

    return tensor, tokenizer