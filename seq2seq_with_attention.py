import tensorflow as tf
import tensorflow_addons as tfa
#from IPython.display import HTML as html_print
#from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import unicodedata
import re
import numpy as np
import os
import io
import time
import wandb
import os
import io
from wandb.keras import WandbCallback
import time
import sys
#from kaggle_secrets import UserSecretsClient
from tensorflow.keras.layers import Embedding, SimpleRNNCell, GRUCell, Dense, LSTMCell

train_file_path = "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
val_file_path= "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
test_file_path  = "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"


class DakshinaDataset:
    def __init__(self, problem_type='en-spa'):
        self.problem_type = 'en-spa'
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None
        self.num_of_train = 0
        self.num_of_test = 0
        self.num_of_val = 0
    

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    ## Step 1 and Step 2 
    def preprocess_sentence(self, w):
        # w = self.unicode_to_ascii(w.lower().strip())

        # # creating a space between a word and the punctuation following it
        # # eg: "he is a boy." => "he is a boy ."
        # # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        # w = re.sub(r"([?.!,¿])", r" \1 ", w)
        # w = re.sub(r'[" "]+', " ", w)

        # # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        # w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '\t' + w + '\n'
        return w
    
    def create_dataset(self, path, data_name):
        # path : path to spa-eng.txt file
        # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
        lines = io.open(path, encoding='UTF-8').read().split('\n')
        #print(lines)
        if data_name == "train":
          self.num_of_train = len(lines) -1
        elif data_name == "val":
          self.num_of_val = len(lines) -1
        else:
          self.num_of_test = len(lines) -1
        word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:len(lines)-1]]
        #print(word_pairs)

        
        return zip(*word_pairs)

    # Step 3 and Step 4
    def tokenize(self, lang):
        # lang = list of sentences in a language
        
        # print(len(lang), "example sentence: {}".format(lang[0]))
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level = True)
        lang_tokenizer.fit_on_texts(lang)

        ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn) 
        ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
        tensor = lang_tokenizer.texts_to_sequences(lang, ) 

        ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences 
        ## and pads the sequences to match the longest sequences in the given input
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self, path,  data_name = None):
        # creating cleaned input, output pairs
        targ_lang, inp_lang ,_= self.create_dataset(path, data_name)
        #print(targ_lang, inp_lang)
        if data_name == "train":
            input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
            #print(input_tensor, inp_lang_tokenizer.word_index)
            target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)
            #print(target_tensor, targ_lang_tokenizer.word_index)
            return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
        else:

            #print(targ_lang, inp_lang)
            input_tensor= self.inp_lang_tokenizer.texts_to_sequences(inp_lang)
            #print(input_tensor)
            input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, padding='post', maxlen = 22)
            #print(input_tensor, inp_lang_tokenizer.word_index)
            target_tensor = self.targ_lang_tokenizer.texts_to_sequences(targ_lang)
            target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, padding='post', maxlen =21)
            #print(target_tensor, targ_lang_tokenizer.word_index)
            return input_tensor, target_tensor

    def call(self, BUFFER_SIZE, BATCH_SIZE):
        file_path = train_file_path
        input_tensor_train, target_tensor_train, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.load_dataset(train_file_path, "train" )
        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        print("val")
        file_path = val_file_path
        input_tensor_val, target_tensor_val = self.load_dataset(val_file_path,  "val")
        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        val_dataset = val_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        print("test")
        file_path = test_file_path
        input_tensor_test, target_tensor_test = self.load_dataset(test_file_path,  "test")
        test_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_test, target_tensor_test))
        test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        # val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        # val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset, test_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer



##### 

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, num_of_layers, enc_unit_type, dropout, recurrent_dropout):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.num_of_layers = num_of_layers
    self.enc_unit_type = enc_unit_type
    self.dropout = dropout
    self.recurrent_dropout = recurrent_dropout
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    ##-------- LSTM layer in Encoder ------- ##
    self.encoder_layer = self.get_encoder_layer(self.enc_units,
                                                self.num_of_layers, self.enc_unit_type)
    
  def get_encoder_layer(self, enc_units, num_of_layers, enc_unit_type):
    return tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells( [self.get_cell(enc_unit_type, 
                                                                                 enc_units) for i in range(num_of_layers)],),
                                  return_sequences=True, return_state=True, name = "Encoder")
  def get_cell(self, cell_type = "lstm", num_of_cell = 1, name = None):
      #print(cell_type)
      if cell_type == "lstm":
        return LSTMCell(num_of_cell, dropout = self.dropout, recurrent_dropout = self.recurrent_dropout, )
      elif cell_type == "rnn":
        return SimpleRNNCell(num_of_cell, dropout = self.dropout, recurrent_dropout = self.recurrent_dropout)
      elif cell_type =="gru":
        return GRUCell(num_of_cell, dropout = self.dropout, recurrent_dropout = self.recurrent_dropout)
      else:
        print(f"Invalid cell type: {cell_type}")

  def call(self, x, hidden):
    x = self.embedding(x)
    output = self.encoder_layer(x, initial_state = hidden)
    return output[0], output[1:]

  def initialize_hidden_state(self):
    if self.enc_unit_type == 'rnn' or self.enc_unit_type == "gru":
        return [tf.zeros((self.batch_sz, self.enc_units))]*self.num_of_layers
    else:
        return [[tf.zeros((self.batch_sz, self.enc_units)),tf.zeros((self.batch_sz, self.enc_units))]]*self.num_of_layers



class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, num_of_layers, dec_unit_type, dropout, recurrent_dropout, attention_type='luong',):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type
    self.num_of_layers = num_of_layers
    self.dec_unit_type = dec_unit_type
    self.dropout = dropout
    self.recurrent_dropout = recurrent_dropout
    # Embedding Layer
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    
    #Final Dense layer on which softmax will be applied
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Define the fundamental cell for decoder recurrent structure
    self.decoder_rnn_cell =  self.get_stacked_rnn_cell()
   


    # Sampler
    self.sampler = tfa.seq2seq.sampler.TrainingSampler()

    # Create attention mechanism with memory = None
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                              None, self.batch_sz*[max_length_input], self.attention_type)

    # Wrap attention mechanism with the fundamental rnn cell of decoder
    self.rnn_cell = self.build_rnn_cell(batch_sz)

    # Define the decoder with respect to fundamental rnn cell
    self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    
  def build_rnn_cell(self, batch_sz):
    rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                  self.attention_mechanism, attention_layer_size=self.dec_units)
    return rnn_cell

  def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
    # ------------- #
    # typ: Which sort of attention (Bahdanau, Luong)
    # dec_units: final dimension of attention outputs 
    # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
    # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

    if(attention_type=='bahdanau'):
      return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
    else:
      return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

  def build_initial_state(self, batch_sz, encoder_state, Dtype):
    decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state


  def call(self, inputs, initial_state):
    x = self.embedding(inputs)
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[max_length_output-1])
    return outputs
  def get_cell(self, cell_type = "lstm", num_of_cell = 1, name = None):
      #print(cell_type)
      if cell_type == "lstm":
        print("encoder cell type = lstm")
        return LSTMCell(num_of_cell, dropout = self.dropout, recurrent_dropout = self.recurrent_dropout, )
      elif cell_type == "rnn":
        print("encoder cell type = rnn")
        return SimpleRNNCell(num_of_cell, dropout = self.dropout, recurrent_dropout = self.recurrent_dropout)
      elif cell_type =="gru":
        print("encoder cell type = gru")
        return GRUCell(num_of_cell, dropout = self.dropout, recurrent_dropout = self.recurrent_dropout)
      else:
        print(f"Invalid cell type: {cell_type}")

  def get_stacked_rnn_cell(self,):
    return tf.keras.layers.StackedRNNCells( [self.get_cell(self.dec_unit_type, self.dec_units,) for i in range(self.num_of_layers)])











class Seq2Seq():
  def __init__(self,vocab_inp_size, vocab_targ_size, encoder_embedding_dim, decoder_embedding_dim, units, batch_sz, num_of_layers, unit_type, dropout, recurrent_dropout, optimizer,metric,attention_type = 'loung'):

    self.encoder = Encoder(vocab_inp_size,encoder_embedding_dim,units,batch_sz,num_of_layers,unit_type,dropout,recurrent_dropout)
    self.decoder = Decoder(vocab_targ_size,decoder_embedding_dim,units,batch_sz,num_of_layers,unit_type,dropout,recurrent_dropout,attention_type )
    self.optimizer = optimizer
    self.metric = metric

  @tf.function
  def val_step(self,inp, targ, enc_hidden):
    loss = 0
    enc_output, enc_state= self.encoder(inp, enc_hidden)


    dec_input = targ[ : , :-1 ] # Ignore <end> token
    real = targ[ : , 1: ]         # ignore <start> token

    # Set the AttentionMechanism object with encoder_outputs
    self.decoder.attention_mechanism.setup_memory(enc_output)

    # Create AttentionWrapperState as initial_state for decoder
    decoder_initial_state = self.decoder.build_initial_state(BATCH_SIZE, tuple(enc_state) ,tf.float32)
    pred = self.decoder(dec_input, decoder_initial_state)
    logits = pred.rnn_output
    loss = self.loss_function(real, logits)
    self.metric.update_state(real, logits)
    return loss, self.metric.result().numpy()


  @tf.function
  def train_step(self,inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
      enc_output, enc_state= self.encoder(inp, enc_hidden)


      dec_input = targ[ : , :-1 ] # Ignore <end> token
      real = targ[ : , 1: ]         # ignore <start> token

      # Set the AttentionMechanism object with encoder_outputs
      self.decoder.attention_mechanism.setup_memory(enc_output)

      # Create AttentionWrapperState as initial_state for decoder
      decoder_initial_state = self.decoder.build_initial_state(BATCH_SIZE, tuple(enc_state) ,tf.float32)
      pred = self.decoder(dec_input, decoder_initial_state)
      logits = pred.rnn_output
      loss = self.loss_function(real, logits)
      self.metric.update_state(real, logits)

    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

    return loss, self.metric.result().numpy()


  def loss_function(self,real, pred):
      # real shape = (BATCH_SIZE, max_length_output)
      # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)  
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss





BUFFER_SIZE = 32000
BATCH_SIZE = 512
# Let's limit the #training examples for faster training
#num_examples = 500

dataset_creator = DakshinaDataset('en-hi')
train_dataset, val_dataset, test_dataset, inp_lang, targ_lang = dataset_creator.call( BUFFER_SIZE, BATCH_SIZE)


example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape


vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim = 256
units = 1024

print("max_length_english, max_length_hindi, vocab_size_english, vocab_size_hindi")
print(max_length_input, max_length_output, vocab_inp_size, vocab_tar_size)


def train(config = None):
    with wandb.init(config = config):
        config = wandb.config

        if config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam()
        else:
            optimizer = tf.keras.optimizers.RMSprop()

        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        seq2seq = Seq2Seq(vocab_inp_size,vocab_tar_size,config.encoder_embedding_dim,config.decoder_embedding_dim,config.unit_size,BATCH_SIZE,config.num_of_layer,config.unit_type,config.dropout,config.recurrent_dropout,optimizer,metric)
        encoder = seq2seq.encoder
        decoder = seq2seq.decoder


        EPOCHS =config.epochs
        tf.config.run_functions_eagerly(True)
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
        step_per_val_epoch = dataset_creator.num_of_val//BATCH_SIZE
        steps_per_epoch = dataset_creator.num_of_train//BATCH_SIZE
        for epoch in range(EPOCHS):
            start = time.time()
            enc_hidden = encoder.initialize_hidden_state()
            total_loss = 0
            total_accuracy = 0
            # print(enc_hidden[0].shape, enc_hidden[1].shape)
            seq2seq.metric.reset_states()
            print("="*80)
            print("TRAINING")
            for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
                batch_loss, batch_acc= seq2seq.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss
                total_accuracy+=batch_acc
                if batch % 10 == 0:
                    #break
                    print('\t Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                          batch,
                                                          batch_loss.numpy(), batch_acc*100 ))

            seq2seq.metric.reset_states()
            total_val_loss = 0
            total_val_accuracy = 0
            print("="*80)
            print("VALIDATING")
            for (batch, (inp, targ)) in enumerate(val_dataset.take(step_per_val_epoch)):
                val_batch_loss, val_batch_acc= seq2seq.val_step(inp, targ, enc_hidden)
                total_val_loss += val_batch_loss
                total_val_accuracy += val_batch_acc
            print(f"Validatiion loss:  {total_val_loss/  step_per_val_epoch}")
            print((f"Validatiion Acc:  {(total_val_accuracy/  step_per_val_epoch)*100}"))

            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print("Accuracy ",(total_accuracy/steps_per_epoch) *100)
            print('Epoch {} Loss {:.4f} Acc {:.4f}'.format(epoch + 1,
                                              total_loss / steps_per_epoch,
                                              (total_accuracy/ steps_per_epoch)*100
                                              ))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            wandb.log({"Epoch": epoch + 1,
                "Train loss": total_loss / steps_per_epoch,
                 "Train Accuracy": (total_accuracy/steps_per_epoch) *100,
                 "Val Accuracy": (total_val_accuracy/  step_per_val_epoch)*100,
                 "Val Loss": total_val_loss/  step_per_val_epoch
                })

sweep_config = {
    
    'method':'random',
    'metric': {
        'name':'Val Accuracy',
        'goal':'maximize'
    },
    'parameters':{
    
    "num_of_layer" : {'values': [1,2,3]},
    "unit_size": {"values":[16,64,256]},
    "unit_type": {"values":["rnn","gru"]},
    "dropout": {"values": [0.0, 0.2, 0.3]},
    'recurrent_dropout':{'values':[0.0,0.2,0.3]},
    "epochs":{"value":15},
    "encoder_embedding_dim":{"values": [64,256, 1024]},
        "decoder_embedding_dim":{"values": [64,256, 1024]},
    "optimizer":{"values": ["rmsprop","adam"]}             
                   }
}

sweep_id = wandb.sweep(sweep_config, project="Sweep_with_Attention2")
wandb.agent(sweep_id, train)
#train()
