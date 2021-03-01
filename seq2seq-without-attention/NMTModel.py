import tensorflow as tf
from util import Util

class NMTModel(tf.keras.Model):
  def __init__(self, vocab, d_model, n_units):
    super(NMTModel, self).__init__()

    self.vocab = vocab
    self.d_model = d_model
    self.n_unit = n_units

    self.encoder = Encoder(vocab.src.size(), d_model, n_units)
    self.decoder = Decoder(vocab.tgt.size(), d_model, n_units)

  def load(self, weights_path):
    optimizer = tf.keras.optimizers.Adam()
    
    ckpt = tf.train.Checkpoint(optimizer=optimizer, 
                                NMTModel=self);
    ckpt.restore(weights_path)
  
  def translate(self, text):
    MAX_LEN = 10
    text = Util.preprocess(text)
    tensor = Util.text_to_sequence([text], self.vocab.src.tokenizer)
    
    enc_output, dec_input, dec_cell = self.encoder(tensor)
    dec_input = tf.expand_dims([self.vocab.tgt.word2index['<start>']], 1)

    translation = []

    initialize_decoder = True
    
    while(dec_input[0][0] != self.vocab.tgt.index2word['<end>'] | len(translation) <= MAX_LEN ):
      prediction, dec_hidden, dec_cell = self.decoder(dec_input, dec_hidden, dec_cell)
      print(prediction)
      dec_input = prediction
      translation.append(prediction[0][0])

    return translation;
    


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, d_model, n_units):
    super(Encoder, self).__init__()

    self.vocab_size = vocab_size + 1
    self.d_model = d_model
    self.n_units = n_units

    self.embedding = tf.keras.layers.Embedding(self.vocab_size, d_model, mask_zero=True)
    self.LSTM = tf.keras.layers.LSTM(n_units, return_sequences = True, return_state = True,
                                     recurrent_initializer='glorot_uniform')

  def call(self, x):
    x = self.embedding(x)
    output, hidden_state, cell_state = self.LSTM(x)
    return output, hidden_state, cell_state

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, d_model, n_units):
    super(Decoder, self).__init__()
    
    self.vocab_size = vocab_size + 1
    self.d_model = d_model
    self.u_units = n_units

    # self.h_projection = tf.keras.layers.Dense(n_units)
    # self.c_projection = tf.keras.layers.Dense(n_units)

    self.embedding = tf.keras.layers.Embedding(self.vocab_size, d_model)

    self.LSTM = tf.keras.layers.LSTM(n_units, return_sequences = True, return_state = True,
                                     recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(self.vocab_size, activation = tf.keras.activations.tanh)

  def call(self, x, h_state, c_state, initialize = False):
    x = self.embedding(x)

    output, hidden_state, cell_state = self.LSTM(x, initial_state = [h_state, c_state])

    x = self.fc(output)
    return x, hidden_state, cell_state