from keras.models import Model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pickle
import tensorflow as tf
from keras.utils.np_utils import to_categorical   


seed = 42
tf.random.set_seed(seed)

batch_size = 64  # Batch size for training.
epochs = 1000  # Number of epochs to train for.
latent_dim = 512  # Latent dimensionality of the encoding space.
num_samples = 6400  # Number of samples to train on.
encoder_shape = 128
decoder_shape = 128
ckpt_every = 10
MODEL_NAME = '6400_model'

def recover_sentence(tokens, dictionary):
    res = []
    key_list = list(dictionary.keys())
    val_list = list(dictionary.values())
    for token in tokens:
        res.append(key_list[val_list.index(token)])
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
# print(recover_sentence(input_texts[0], input_characters))
# input("Press enter to continue training.")
# exit()
en_pad_sentence = pad_sequences(input_texts, max_encoder_seq_length, padding = "post")
zh_pad_sentence = pad_sequences(target_texts, max_decoder_seq_length, padding = "post")
# Reshape data
en_pad_sentence = en_pad_sentence.reshape(*en_pad_sentence.shape, 1)
zh_pad_sentence = zh_pad_sentence.reshape(*zh_pad_sentence.shape, 1)

# print(zh_pad_sentence[0])
# print(zh_pad_sentence[0].shape)
# exit()
if(MODEL_NAME != None):
    model = tf.keras.models.load_model(MODEL_NAME)
    print(f"Loaded {MODEL_NAME}")
else:
    input_sequence = Input(shape=(max_encoder_seq_length))
    embedding = Embedding(input_dim=num_encoder_tokens, output_dim=latent_dim,)(input_sequence)
    encoder = LSTM(encoder_shape, return_sequences=False)(embedding)
    r_vec = RepeatVector(max_decoder_seq_length)(encoder)
    decoder = LSTM(decoder_shape, return_sequences=True, dropout=0)(r_vec)
    logits = TimeDistributed(Dense(num_decoder_tokens))(decoder)
    model = Model(input_sequence, Activation('softmax')(logits))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=Adam(1e-3),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# model.summary()
# X_train, X_test, y_train, y_test = train_test_split(en_pad_sentence, zh_pad_sentence, test_size = 0.3, random_state = 42)
# filepath="ckpt/weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
model.summary()
X_train, X_test, y_train, y_test = train_test_split(en_pad_sentence, zh_pad_sentence, test_size = 0.3, random_state = 42)
filepath="ckpt/weights-improvement-{epoch:02d}-{sparse_categorical_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='sparse_categorical_accuracy', verbose=1, save_best_only=True, mode='max', save_freq= int(ckpt_every * len(X_train) / batch_size))

callbacks_list = [checkpoint]
model_results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list)
# print(model_results.history)
model.save(str(num_samples) + '_model')
print('[Test loss, Test accuracy]: ', model.evaluate(X_test, y_test, batch_size = batch_size))

# for idx, prediction in enumerate(range(X_test.shape[0])):
#     prediction = model.predict(X_test[idx: idx+1], batch_size = batch_size)[0]
#     print(prediction)
#     print('English sentence: ')
#     print('\t' + ' '.join(recover_sentence(X_test[idx], input_characters)))
#     print('Correct translation:')
#     print('\t' + ''.join(recover_sentence(y_test[idx], target_characters)))
#     print('Translated as:')
#     print('\t' + ''.join(recover_prediction(prediction, target_characters)))
#     print('-' * 50)