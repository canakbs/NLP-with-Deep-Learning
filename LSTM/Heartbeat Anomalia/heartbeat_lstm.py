import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight # veri dengeli olmadığı 
#için öğrenmede yatkınlık(bias) oluşabilir 
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


#define what the data is and how we will use it?

# normal:0 anomalia:1
normal_path=r"C:\Users\LENOVO\OneDrive\Masaüstü\Çalışmalar\NLP-with-Deep-Learning\LSTM\Heartbeat Anomalia\ptbdb_normal.csv"
abnormal_path=r"C:\Users\LENOVO\OneDrive\Masaüstü\Çalışmalar\NLP-with-Deep-Learning\LSTM\Heartbeat Anomalia\ptbdb_abnormal.csv"
normal = pd.read_csv(normal_path, header=None)
abnormal = pd.read_csv(abnormal_path, header=None)

normal["label"] = 0
abnormal["label"] = 1

df = pd.concat([normal, abnormal], axis=0)
df = df.sample(frac=1).reset_index(drop=True)  # shuffle

#look at closer



print(df.head()) # values are normalized
print(df.shape)
print(df.info())

labels = ["Normal", "Anomaly"]
sizes = [
    (df["label"] == 0).sum(),
    (df["label"] == 1).sum()
]

plt.figure(figsize=(6,6))
plt.pie(sizes,labels=labels)
plt.title("Class Distribution")
plt.axis("equal")
plt.show()


normal_sample = df[df.label == 0].iloc[0, :-1]
abnormal_sample = df[df.label == 1].iloc[0, :-1]

plt.figure(figsize=(12,4))
plt.plot(normal_sample, label="Normal")
plt.plot(abnormal_sample, label="Abnormal")
plt.legend()
plt.title("Single ECG Beat Comparison")
plt.show()



X = df.iloc[:, :-1].values   # (samples, 187)
y = df["label"].values

X = X.reshape(X.shape[0], X.shape[1], 1)
# (samples, timesteps, features)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight = dict(enumerate(class_weights))
print(class_weight)

model = Sequential()

model.add(
    Bidirectional(
        LSTM(64, return_sequences=True),
        input_shape=(187, 1)
    )
)
model.add(Dropout(0.3))

model.add(
    Bidirectional(
        LSTM(32)
    )
)
model.add(Dropout(0.3))

model.add(Dense(1, activation="sigmoid"))


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.15,
    class_weight=class_weight,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    class_weight=class_weight,
    verbose=1
)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:}")



y_pred = (model.predict(X_test) > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))

model.save("heartbeatLSTM.h5")
print("Model kaydedildi")



