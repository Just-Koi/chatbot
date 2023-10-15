import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json 
import pickle

intents_file = open('intents.json').read()
intents = json.loads(intents_file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', ';', ':']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        # Add documents
        documents.append((wrds, intent['tag']))
        # Add to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)

# lemmatize the words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

# sort
classes = sorted(list(set(classes)))

# combination between patterns and intents
print (len(documents), "documents")

# intents
print (len(classes), "classes", classes)

# vocabulary
print (len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# create the training data
training = []
# empty array for the output
output = [0] * len(classes)
# training set
for doc in documents:
    # bag of words
    bag = []
    # list of tokenized words
    word_patterns = doc[0]
    # lemmanize each word
    bag = np.array([1 if lemmatizer.lemmatize(word.lower()) in word_patterns else 0 for word in words])
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle our features and turn into numpy arrays
random.shuffle(training)
training = np.array(training, dtype="object")
# create training and testing sets
training_data = np.array(training)
x_train = np.array(training_data[:,0].tolist(), dtype=np.float32)
y_train = np.array(training_data[:,1].tolist(), dtype=np.float32)

print("Training data is created")

# deep neural networks model
model = Sequential()
model.add(Dense(128, input_dim=len(x_train[0]), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# training the model and saving the model
mod = model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', mod)

print("Model is saved")