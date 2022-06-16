import json, ast
#from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

print("Name of file to train on:")
file_name = input()
#old data pulled out
file_data = []
#text data
texts = []
#art data, instead of trying to combine text and art into one input, use multiple input and input art data at coconinate
art = []
#labels
labels = []


#pull file data and place as dictionaries in list
with open(file_name) as file:
    for line in file:
        data_line = ast.literal_eval(line.rstrip())
        file_data.append(data_line)

file.close()
#print(data[0]["text"])

#convert text to word sequence
for item in file_data:
    if item["voted_up"]:
        labels.append(1)
    else:
        labels.append(0)
    if item["art_present"]:
        art.append(1)
    else:
        art.append(0)
    texts.append(item["text"])
    

maxlen = 500
training_samples = 8000
validation_samples = 2000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
#print("sequences:")
#print(sequences)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
#print("data")
#print(data)
labels = np.asarray(labels)
art = np.asarray(art)
print('Shape of text data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print('Shape of art data tensor: ', art.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
art = art[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
z_train = art[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
z_val = art[training_samples: training_samples + validation_samples]

print(x_train.shape)

#plan:
#push text data into 1d convnent and use concononate layer to merge art data and convnent output
#current config needs only 30 epochs to hit 90% accuracy

from keras.models import Sequential
from keras import layers
#error when imported from keras
#from keras.optimizers import RMSprop
from tensorflow.keras.optimizers import RMSprop
#for functional model type
from keras.models import Model
from keras import Input

max_features = 50000

# Model version one, single text input
#model = Sequential()
#model.add(layers.Embedding(max_features, 128, input_length=maxlen))
#model.add(layers.Conv1D(32, 7, activation='relu'))
#model.add(layers.MaxPooling1D(5))
#model.add(layers.Conv1D(32, 7, activation='relu'))
#model.add(layers.GlobalMaxPooling1D())
#model.add(layers.Dense(1))

# Model version two, text and art input
# text input
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(max_features, 128)(text_input)
conv1d_text = layers.Convolution1D(32, 7, activation='relu')(embedded_text)
maxpool1d_text = layers.MaxPooling1D(5)(conv1d_text)
conv1d_2_text = layers.Convolution1D(32, 7, activation='relu')(maxpool1d_text)
text_layers = layers.GlobalMaxPooling1D()(conv1d_2_text)

# art input
art_shape = 1
art_input = Input(shape=(art_shape,))

# merge
concat_layer= layers.Concatenate()([text_layers, art_input])
output = layers.Dense(2)(concat_layer)
output = layers.Dense(1)(output)

model = Model(inputs=[text_input, art_input], outputs=output)


model.summary()

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit([x_train, z_train], y_train,
                    epochs=40,
                    batch_size=128,
                    validation_split=0.2)

#graph history
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.figure(1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


#plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.figure(2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.show()
#model.save("./Saved Model v2")