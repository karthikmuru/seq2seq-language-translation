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
    MAX_LEN = 50
    text = Util.preprocess(text)
    tensor = Util.text_to_sequence([text], self.vocab.src.tokenizer)
    
    o_enc, h_enc, c_enc = self.encoder(tensor)
    dec_input = tf.expand_dims([self.vocab.tgt.word2index['<start>']], 1)

    translation = []
    
    while(dec_input[0][0] != self.vocab.tgt.index2word['<end>'] | len(translation) <= MAX_LEN ):
      prediction, _, _ = self.decoder(dec_input, h_enc, c_enc, o_enc)
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
    self.n_units = n_units

    self.attention = Attention(n_units)

    # self.h_projection = tf.keras.layers.Dense(n_units)
    # self.c_projection = tf.keras.layers.Dense(n_units)

    self.embedding = tf.keras.layers.Embedding(self.vocab_size, d_model)

    self.LSTM = tf.keras.layers.LSTM(n_units, return_sequences = True, return_state = True,
                                     recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.n_units, activation = tf.keras.activations.tanh, input_shape=(None, 2*self.n_units))
    self.fc2 = tf.keras.layers.Dense(self.vocab_size, activation = tf.keras.activations.tanh)

  def call(self, x, h_state, c_state, h_enc_output):
    x = self.embedding(x)

    o_dec, h_dec, c_dec = self.LSTM(x)

    a = self.attention(h_enc_output, h_dec)

    u_t = tf.concat([a, tf.squeeze(o_dec)], axis=1)
    # u_t -> (batch_size, 2*n_units)

    o_t = self.fc1(u_t)
    # o_t -> (batch_size, n_units)

    p_t = self.fc2(o_t)
    # p_t -> (batch_size, tgt_vocab_size)

    return p_t, h_dec, c_dec

class Attention(tf.keras.layers.Layer):
  def __init__(self, n_units):
    super(Attention, self).__init__()

    self.n_units = n_units
    self.attention_projection = tf.keras.layers.Dense(n_units)

  def call(self, h_enc, h_dec):
    # h_enc -> (batch_size, seq_len, n_units)
    h_enc = self.attention_projection(h_enc)
    
    # h_dec -> (batch_size, n_units)
    h_dec = tf.expand_dims(h_dec, 1)
    # h_dec -> (batch_size, 1, n_units)

    alpha = tf.squeeze(tf.linalg.matmul(h_enc, h_dec, transpose_b=True))
    # alpha -> (batch_size, seq_len)

    attention = tf.squeeze(tf.linalg.matmul(tf.expand_dims(alpha, 1), h_enc))
    # attention -> (batch_size, n_units)
    
    return attention