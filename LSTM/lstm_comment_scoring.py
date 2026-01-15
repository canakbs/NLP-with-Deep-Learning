# predict the point between 1-10 based on customers comments

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import pickle  # tokenizer kaydetme için
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ===============================
# DATASET (HATASIZ YÜKLEME)
# ===============================
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
df = dataset["train"].to_pandas()

print(df.head())

# ===============================
# LABEL DÜZENLEME (REGRESSION)
# ===============================
df["label"] = df["label"] + 1   # 1–5 arası skor

texts = df["text"].values
labels = df["label"].values

# ===============================
# TOKENIZER
# ===============================
tokenizer = Tokenizer(
    num_words=10000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts)

# tokenizer kaydet
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)

# ❗ HATA DÜZELTİLDİ
sequences = tokenizer.texts_to_sequences(texts)

padded_sequences = pad_sequences(
    sequences,
    maxlen=100,
    padding="post",
    truncating="post"
)

# ===============================
# LABEL NORMALIZATION (REGRESSION)
# ===============================
scaler = MinMaxScaler()
labels_scaled = scaler.fit_transform(labels.reshape(-1, 1))

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    labels_scaled,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_train[:2])



model=Sequential()

model.add(Embedding(input_dim=10000,output_dim=128,input_length=100))

model.add(LSTM(128))#128 hücre

model.add(Dense(64,activation="relu"))
model.add(Dense(1, activation="linear"))

model.compile(optimizer="adam",
              loss=MeanSquaredError(),
              metrics=[MeanAbsoluteError()])

print(model.summary())


history=model.fit(
    X_train,y_train,batch_size=64,
    validation_split=0.2
)
plt.figure(figsize=(8,6))
plt.plot(history.history["loss"],label="Trainnig Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.title("Eğitim süreci")
plt.xlabel("Epochs")
plt.ylabel("Loss MSE")
plt.grid()
plt.legend()
plt.show()

model.save("predict_comment_lstm.h5")
print("Model saved")