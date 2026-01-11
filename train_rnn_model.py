"""
Create a virtual environment
 Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
>>
 .\venv\Scripts\activate

"""
#Sentiment Analyis using RNN
# RNN: Önceki girdileri öğrenerek sonraki kelimeler için tahmin yapar
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
stop_words=set(stopwords.words("english"))

max_features=10000
imdb.load_data(number_words=max_features)

(X_train, y_train), (X_test, y_test)=imdb.load_data(num_words=max_features)

original_word_index=imdb.get_word_index()
inv_word_index={value+3:key for key,value in original_word_index.items()}

inv_word_index[0]="<PAD>"
inv_word_index[1]="<START>"
inv_word_index[2]="<UNK>"

def decode_review(text):
    return " ".join([inv_word_index.get(i,"?") for i in text])


print(X_train[0])

print(decode_review(X_train[0]))

print(y_train[0])



def preprocess_reviews(reviews):
    processed_reviews = []

    for review in reviews:          # review = [1, 14, 22, ...]
        filtered_review = [
            idx for idx in review   # idx = 14
            if inv_word_index.get(idx, "") not in stop_words
        ]
        processed_reviews.append(filtered_review)

    return processed_reviews

 
X_train=preprocess_reviews(X_train)
X_test=preprocess_reviews(X_test)  

#Neural networkler aynı uzunlukta verilerle çalışır 
#Pad sequence: Uzunluk eşitleme: kısa metinlere 0 ekler, uzun metinleri kısaltır.

maxlen=500
X_train=pad_sequences(X_train, maxlen=maxlen,)
X_test=pad_sequences(X_test, maxlen=maxlen)



#RNN MODEL
#iskelet
model=Sequential()
#embedding katmanı: kelimeleri 32 boyutlu vektörlere dönüştürür
model.add(Embedding(input_dim=max_features, output_dim=32, input_length=maxlen))
#RNN katmanı: 32 birimlik RNN
model.add(SimpleRNN(units=32)) # nöron sayısı
#binary sınıflandırma için çıktı katmanı
model.add(Dense(1, activation='sigmoid'))# son katman
#sigmoid function: 0 veya 1 çıktı verir

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#adam optimizer: ağırlık(weight and bias) güncelleme yöntemi
# binary_crossentropy loss: binary sonuçlardaki hata ölçümü
#accuracy metriği: doğruluk ölçümü
print(model.summary())
# RNN de bilgile eksik kalabiliyor


#Training
history=model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2)
#epochs: tüm veri setinin kaç kez işleneceği
#batch_size: her seferde kaç örneğin işleneceği
#validation_split: eğitim verisinin %20'si doğrulama için ayrılır


#Visualization

def plot_history(history):
    #accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epochs')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    
   
    #loss
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"],label="Training Loss")
    plt.plot(history.history["val_loss"],label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.tight_layout()
    plt.grid()
    plt.show()

plot_history(history)

test_loss,test_acc=model.evaluate(X_test,y_test)
print("Test Results")
print("Test Loss:",test_loss)   
print("Test Accuracy:",test_acc)

model.save("rnn_sentiment_analysis_model.h5")
print("Model saved as rnn_sentiment_analysis_model.h5")










