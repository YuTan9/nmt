from operator import index
from keras.models import Model, load_model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import keras
from tqdm import tqdm
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 512  # Latent dimensionality of the encoding space.
num_samples = 6400  # Number of samples to train on.
encoder_shape = 128
decoder_shape = 128
MODEL_NAME = 'ckpt/weights-improvement-1750-0.92.hdf5'

def recover_sentence(tokens, dictionary):
    res = []
    key_list = list(dictionary.keys())
    val_list = list(dictionary.values())
    for token in tokens:
        if(token[0] == 0):
            break
        res.append(key_list[val_list.index(token[0])])
    return res

def recover_prediction(token_onehots, dictionary):
    res = []
    key_list = list(dictionary.keys())
    val_list = list(dictionary.values())
    for token_onehot in token_onehots:
        index = np.argmax(token_onehot)
        if(index == 0):
            break
        res.append(key_list[val_list.index(index)])

    return res

# Vectorize the data.
input_texts = np.load('data/en' + str(num_samples) + '.npy', allow_pickle = True)
target_texts = np.load('data/zh' + str(num_samples) + '.npy', allow_pickle = True)
with open('data/en' + str(num_samples) + '_dict.pkl', 'rb') as f:
    input_characters = pickle.load(f)
with open('data/zh' + str(num_samples) + '_dict.pkl', 'rb') as f:
    target_characters = pickle.load(f)
num_encoder_tokens = max(input_characters.values()) + 1
num_decoder_tokens = max(target_characters.values()) + 1
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
# avg_encoder_seq_length = sum([len(txt) for txt in input_texts]) / num_samples
# avg_decoder_seq_length = sum([len(txt) for txt in target_texts]) / num_samples
# print('Average sequence length for inputs:', avg_encoder_seq_length)
# print('Average sequence length for outputs:', avg_decoder_seq_length)
# # print(recover_sentence(input_texts[0], input_characters))
# exit()
en_pad_sentence = pad_sequences(input_texts, max_encoder_seq_length, padding = "post", value = 0)
zh_pad_sentence = pad_sequences(target_texts, max_decoder_seq_length, padding = "post", value = 0)

# Reshape data
en_pad_sentence = en_pad_sentence.reshape(*en_pad_sentence.shape, 1)
zh_pad_sentence = zh_pad_sentence.reshape(*zh_pad_sentence.shape, 1)

model = keras.models.load_model(MODEL_NAME)
# model = keras.models.load_model('weights')
model.summary()
# exit()
X_train, X_test, y_train, y_test = train_test_split(en_pad_sentence, zh_pad_sentence, test_size = 0.3, random_state = 42)

# exit()
# print('[Test loss, Test accuracy]: ', model.evaluate(X_test, y_test, batch_size = batch_size))
# [Test loss, Test accuracy]:  [0.6097457408905029, 0.9301545023918152]

# a = model.predict(X_test[0:2], batch_size = batch_size)
# # b = model.predict(X_test[1: 2], batch_size = batch_size)[0]
# # print(X_test[0: 1])
# # print(X_test[0: 1] == X_test[1: 2])
# print((a[0] == a[1]).all())
# exit()
output = ""
accuracy = []
for idx in range(X_test.shape[0]):
    m = tf.keras.metrics.SparseCategoricalAccuracy()
    prediction = model.predict(X_test[idx: idx+1], batch_size = 1, verbose = 0)[0]
    m.update_state(y_test[idx], prediction)
    output += 'accuracy: ' + str(m.result().numpy()) + "\n"
    output += 'English sentence: '
    output += '\t' + ' '.join(recover_sentence(X_test[idx], input_characters))
    output += 'Correct translation:'
    output += '\t' + ''.join(recover_sentence(y_test[idx], target_characters))
    output += 'Translated as:'
    output += '\t' + ''.join(recover_prediction(prediction, target_characters))
    output += '-' * 50
    accuracy.append(m.result().numpy())
    if(m.result().numpy() >= 0.95):
        # print(prediction)
        print(idx)
        print('English sentence: ')
        print('\t' + ' '.join(recover_sentence(X_test[idx], input_characters)))
        print('Correct translation:')
        print('\t' + ''.join(recover_sentence(y_test[idx], target_characters)))
        print('Translated as:')
        print('\t' + ''.join(recover_prediction(prediction, target_characters)))
        print('-' * 50)
        # input()
print("Finished, press enter")
input()
print(output)
_ = plt.hist(accuracy, bins='auto')
plt.savefig('hist_5000epoch.png')