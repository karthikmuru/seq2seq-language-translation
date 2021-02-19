import numpy as np
import tensorflow as tf
import re
import io
import os
import time

from util import Util
from vocab import Vocab
from NMTModel import NMTModel

inp_path = './data/train.tags.de-en.de'
targ_path = './data/train.tags.de-en.en'
d_model = 256
n_units = 1024
batch_size = 128
EPOCHS = 10

# Set this as None to train on full dataset
NUM_EXAMPLES = 1000

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

@tf.function
def train_step(input, target):
  loss = 0
  
  with tf.GradientTape() as tape:
    enc_output, dec_hidden, dec_cell = NMT.encoder(input)

    dec_input = tf.expand_dims([NMT.vocab.tgt.word2index['<start>']] * 128, 1)

    initialize_decoder = True
    for i in range(1, target.shape[1]):
      predictions, dec_hidden, dec_cell = NMT.decoder(dec_input, dec_hidden, dec_cell, initialize_decoder)
      loss += loss_function(target[:, i], predictions)
      dec_input = tf.expand_dims(target[:, i], 1)
      initialize_decoder = False
  
  batch_loss = (loss / int(target.shape[1]))
  variables = NMT.encoder.trainable_variables + NMT.decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def train_NMT():
  for epoch in range(EPOCHS):
    start = time.time()

    total_loss = 0

    for (batch, (input, target)) in enumerate(dataset.take(steps_per_epoch)):
      batch_loss = train_step(input, target)

      total_loss += batch_loss

      if batch % 1 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

    if (epoch + 1) % 2 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

input, input_tokenizer = Util.load_dataset(inp_path, NUM_EXAMPLES)
target, target_tokenizer = Util.load_dataset(targ_path, NUM_EXAMPLES)

steps_per_epoch = len(input) // batch_size

dataset = tf.data.Dataset.from_tensor_slices((input, target)).shuffle(len(input))
dataset = dataset.batch(batch_size)

optimizer = tf.keras.optimizers.Adam()
vocab = Vocab.build(inp_path, targ_path, NUM_EXAMPLES)
vocab.save('./weights/vocab.json')

print(len(vocab.src.index2word))
print(len(vocab.tgt.index2word))

NMT = NMTModel(vocab, d_model, n_units)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, NMTodel=NMT)

print('Training NMT model...')
print(input.shape)
train_NMT()