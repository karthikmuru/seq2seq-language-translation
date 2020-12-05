import tensorflow as tf
import tensorflow.keras.layers as layers

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import numpy as np
import os
import io
import time

from util import *

import sys
matplotlib.use('agg')
sys.path.insert(0, './weights') 
weights_dir = './model/weights/'

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, d_model, n_units):
    super(Encoder, self).__init__()
    self.n_units = n_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    self.gru = tf.keras.layers.GRU(self.n_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    print(f"Encoder input x shape {x.shape}")
    x = self.embedding(x)
    print(f"Encoder embedding shape {x.shape}")
    output, state = self.gru(x, initial_state = hidden)
    print(f"Encoder GRU output {output.shape}")
    print(f"Encoder GRU state {state.shape}")
    return output, state

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, d_model, dec_units):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)
    # print(f"Decoder x : {x.shape}")

    x = self.embedding(x)
    
    # print(f"Decoder x embedding : {x.shape}")
    # print(f"Decoder context_vector: {context_vector.shape}")
    # print(f"Decoder attention_weights: {context_vector.shape}")

    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # print(f"Decoder x + attention_weights: {context_vector.shape}")
    output, state = self.gru(x)

    # print(f"Decoder GRU output : {output.shape}")
    # print(f"Decoder GRU state : {state.shape}")

    output = tf.reshape(output, (-1, output.shape[2]))

    # print(f"Decoder GRU output after reshape : {output.shape}")
    x = self.fc(output)

    print(f"Decoder Dense layer : {x.shape}")
    return x, state, attention_weights


class TranslationModel():
  def __init__(self):
    self.d_model = 256
    self.units = 1024

    self.inp_lang_index_word = np.load(weights_dir + 'inp_lang_index_word.npy', allow_pickle=True).all()
    self.inp_lang_word_index = np.load(weights_dir + 'inp_lang_word_index.npy', allow_pickle=True).all()
    self.targ_lang_index_word = np.load(weights_dir + 'targ_lang_index_word.npy', allow_pickle=True).all()
    self.targ_lang_word_index = np.load(weights_dir + 'targ_lang_word_index.npy', allow_pickle=True).all()
    self.model_weights_location = weights_dir + 'ckpt-5'

    
    self.vocab_inp_size = len(self.inp_lang_word_index) + 1
    self.vocab_targ_size = len(self.targ_lang_word_index) + 1
    
    self.encoder = Encoder(self.vocab_inp_size, self.d_model, self.units)
    self.decoder = Decoder(self.vocab_targ_size, self.d_model, self.units)
    self.optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(optimizer=self.optimizer,
                                 encoder=self.encoder,
                                 decoder=self.decoder)
    ckpt.restore(self.model_weights_location)

  def load_checkpoint(self):
    self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer,
                                 encoder=self.encoder,
                                 decoder=self.decoder)
    self.ckpt.restore(self.model_weights_location)

  def evaluate(self, sentence):
    attention_plot = np.zeros((51, 51))

    sentence = preprocess_sentence(sentence)

    inputs = [self.inp_lang_word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                          maxlen=51,
                                                          padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, self.units))]
    enc_out, enc_hidden = self.encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([self.targ_lang_word_index['<start>']], 0)

    for t in range(51):
      predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                          dec_hidden,
                                                          enc_out)

      # storing the attention weights to plot later on
      # print(f"attention weights plot {attention_weights}")

      attention_weights = tf.reshape(attention_weights, (-1, ))

      # print(f"attention weights plot {attention_weights}")
      attention_plot[t] = attention_weights.numpy()

      predicted_id = tf.argmax(predictions[0]).numpy()

      result += self.targ_lang_index_word[predicted_id] + ' '

      if self.targ_lang_index_word[predicted_id] == '<end>':
        return result, sentence, attention_plot

      # the predicted ID is fed back into the model
      dec_input = tf.expand_dims([predicted_id], 0)
  
  def plot_attention(self, attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig('attention.png')

  def translate(self, sentence):
    result, sentence, attention_plot = self.evaluate(sentence)
    
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    attention = self.plot_attention(attention_plot, sentence.split(' '), result.split(' '))
    
    return result

translation_model = TranslationModel();