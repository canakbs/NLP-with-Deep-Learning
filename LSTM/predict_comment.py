import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.sequence import pad_sequences

model=load_model("predict_comment_lstm.h5",compile=False)


with open("tokenizer.pkl","rb") as f:
    tokenizer=pickle.load(f)

#2 star
data=["Unfortunately, the frustration of being Dr. Goldberg's patient is a repeat of the experience I've had with so many other doctors in NYC -- good doctor, terrible staff. It seems that his staff simply never answers the phone. It usually takes 2 hours of repeated calling to get an answer. Who has time for that or wants to deal with it? I have run into this problem with many other doctors and I just don't get it. You have office workers, you have patients with medical needs, why isn't anyone answering the phone? It's incomprehensible and not work the aggravation. It's with regret that I feel that I have to give Dr. Goldberg 2 stars."]
sequences=tokenizer.texts_to_sequences(data)

padded=pad_sequences(sequences,maxlen=100,padding="post")

prediction=model.predict(padded)

prediction_scaled=prediction*5

print(f"{data}\n {prediction_scaled}")