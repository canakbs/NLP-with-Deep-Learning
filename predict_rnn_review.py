import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

max_features=10000
maxlen=500
nltk.download('stopwords')
stop_words=set(stopwords.words("english"))

original_word_index=imdb.get_word_index()
index_to_word={value+3:key for key,value in original_word_index.items()}
index_to_word[0]="<PAD>"
index_to_word[1]="<START>"
index_to_word[2]="<UNK>"

#ters sözlük
word_to_index={v:k for k,v in index_to_word.items()} 

model=load_model("rnn_sentiment_analysis_model.h5")
print("Model yüklendi.")

def predict_review(model, review):
    review = review.lower()
    words = text_to_word_sequence(review)

    encoded = []
    for word in words:
        if word not in stop_words and word in original_word_index:
            idx = original_word_index[word] + 3
            if idx < max_features:
                encoded.append(idx)
            else:
                encoded.append(2)  # UNK
        else:
            encoded.append(2)      # UNK

    padded = pad_sequences([encoded], maxlen=maxlen)

    prediction = model.predict(padded, verbose=0)[0][0]
    print("Raw prediction:", prediction)

    return "Pozitif" if prediction >= 0.5 else "Negatif"

print("Lütfen bir film yorumu girin:")
input_review=input()
sentiment=predict_review(model, input_review)
print(f"Tahmin: {sentiment}")