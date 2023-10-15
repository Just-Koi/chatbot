import nltk 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model.h5')
import json
import random

intents_file = open('intents.json').read()
intents = json.loads(intents_file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

print(intents_file)  # Print the content of the intents file
print(intents)       # Print the loaded intents data

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    # filter below threshold predictions
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    print("Predictions:", res)  # Print predictions for debugging
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Make sure there are results before sorting
    if results:
        # sorting strength probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            # Check if the index is within the range of classes list
            if r[0] < len(classes):
                return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
            else:
                print("Index out of range error. r[0]:", r[0])
        return return_list
    else:
        # Handle the case when there are no results above the threshold
        print("No intents detected above the threshold.")
        return None

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Tkinrker GUI
import tkinter
from tkinter import *

root = Tk()
root.title("Chatbot")
root.geometry("500x500")
root.resizable(width=True, height=True)

# Create chat window
ChatBox = Text(root, bd=0, height="8", width="50", font="Roboto")
ChatBox.config(state=DISABLED)

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg.strip():
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#EEE5E9", font=('Roboto', 13))

        ints = predict_class(msg)

        if ints:  # Check if intents are detected
            tag = ints[0]['intent']
            res = getResponse(ints, intents)
        else:
            tag = "unknown"
            res = "I'm sorry, I don't understand that."

        ChatBox.insert(END, "Bot: " + res + '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)

# Bind scrollbar 
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Create button to send message
SendButton = Button(root, height="5", text="Send", command=send, font="Roboto", bg="#92DCE5", activebackground="#75b0b7", fg="#EEE5E9", cursor="hand2")

# Create the box to enter message
EntryBox = Text(root, bd=0, bg="#2B303A", fg="#EEE5E9", height="5", width="29", font="Roboto")
EntryBox.bind("<Return>", send)

# Place all components in the window
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()