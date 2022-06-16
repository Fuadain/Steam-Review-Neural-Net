import string, nltk, keras
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.optimizers import RMSprop

def cleanUp(text):
    data = {"text":text, "art_present":0}

    #detect ascii art
    if '⣿' in text:
        data["art_present"] = 1

    # remove UNICODE characters
    text_encode = text.encode(encoding="ascii", errors="ignore")
    text = text_encode.decode()

    # cleaning the text to remove extra whitespace 
    text = " ".join([word for word in text.split()])

    # remove punctuation
    punct = set(string.punctuation) 
    text = "".join([ch for ch in text if ch not in punct])

    # make text lower case
    text = text.lower()
    
    # remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    data["text"] = text
    return data

print("Input NN model you would like to use: (Enter 1 or 2)" + 
      "\n1) Version 1 (doesn't consider art used for steam reviews made from UNICODE braille)" +
      "\n2) Version 2 (does consider art used for steam reviews made from UNICODE braille)")
model_input = input()

print("Enter review text for NN model to predict if it is a positive or negative review:")
text_input = input()

# Clean and prepare text data
text_data = cleanUp(text_input)
text = text_data["text"]
text = [text]

# Text preprocessing
maxlen = 500
max_words = 500

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

sequences = pad_sequences(sequences, maxlen=maxlen)
prediction_input = sequences

# Choose which model
if model_input == "1":
    model = keras.models.load_model('./Saved Model')
else:
    model = keras.models.load_model('./Saved Model v2')
    # prepare art data
    art_input = [[text_data["art_present"]]]
    art_input = np.asarray(art_input)
    print("art input shape: ", art_input.shape)

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

#prediction
if model_input == "1":
    # Model v1
    predict = model.predict(prediction_input)
else:
    # Model v2
    predict = model.predict([prediction_input, art_input])

print(predict)

print("\nPrediction:")
if predict > 0.5:
    print("Positive review")
else:
    print("Negative review")
