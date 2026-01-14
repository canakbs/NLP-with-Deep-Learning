import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


data = [
    "Bugün hava çok güzel",
    "Sabah erkenden kalkıp yürüyüş yaptım",
    "Dün izlediğim film beni çok etkiledi",
    "Bu kitabı okurken zamanın nasıl geçtiğini anlamadım",
    "Kahvemi içerken müzik dinlemeyi seviyorum",
    "Yoğun bir günün ardından eve gelmek çok iyi hissettirdi",
    "Bugün işler planladığım gibi gitmedi",
    "Yeni bir şeyler öğrenmek beni motive ediyor",
    "Uzun zamandır böyle mutlu hissetmemiştim",
    "Sınav sonuçlarını görünce biraz hayal kırıklığı yaşadım",
    "Arkadaşlarımla vakit geçirmek bana iyi geliyor",
    "Bugün kendimi çok yorgun hissediyorum",
    "Bu proje üzerinde çalışmak bana çok şey kattı",
    "Yağmurlu havalar bazen insanı düşündürüyor",
    "Yeni hedefler belirlemek beni heyecanlandırıyor",
    "Sabırsızlıkla hafta sonunu bekliyorum",
    "Bu dersi anlamak düşündüğümden zor oldu",
    "Uzun bir aradan sonra tekrar spor yapmaya başladım",
    "Bugün her şey üst üste geldi",
    "Kendime biraz zaman ayırmam gerektiğini fark ettim",
    "Bu uygulama beklediğimden çok daha kullanışlı",
    "Son zamanlarda kendimi geliştirmeye odaklandım",
    "Yeni tanıştığım insanlar bana farklı bakış açıları kazandırdı",
    "Bazen küçük şeyler bile mutlu olmaya yetiyor",
    "Bu haberi duyunca gerçekten çok şaşırdım",
    "Planlarımın iptal olması moralimi bozdu",
    "Bugün oldukça verimli bir gündü",
    "Uzun süredir ertelediğim işi sonunda bitirdim",
    "Hatalarımdan ders çıkarmaya çalışıyorum",
    "Yeni bir başlangıç yapmak istiyorum",
    "Bugün kendime daha fazla güvendim",
    "Yoğun tempoya alışmak biraz zaman alıyor",
    "Bu şehirde yaşamayı seviyorum",
    "Küçük başarılar büyük motivasyon sağlıyor",
    "Bugün beklenmedik güzel bir haber aldım",
    "Bazen her şeyden uzaklaşmak gerekiyor",
    "Bu konuyu daha detaylı öğrenmek istiyorum",
    "Zor zamanlar insanı daha güçlü yapıyor",
    "Bugün kendimle gurur duydum",
    "Yeni bir dil öğrenmek sabır gerektiriyor",
    "Bu hatayı bir daha tekrarlamak istemiyorum",
    "Bugün oldukça stresli geçti",
    "Hedeflerime adım adım yaklaşıyorum",
    "Kendimi daha disiplinli hissetmeye başladım",
    "Bu deneyim bana çok şey öğretti",
    "Bugün motivasyonum oldukça yüksekti",
    "Zamanı daha iyi kullanmam gerektiğini fark ettim",
    "Bu yolculuk benim için çok anlamlıydı",
    "Bugün kendimi oldukça enerjik hissediyorum",
    "Uzun süredir planladığım projeye sonunda başladım",
    "Sabah haberlerini izlemek moralimi biraz bozdu",
    "Yeni fikirler üretmek beni heyecanlandırıyor",
    "Bugün yapılacaklar listemi tamamen bitirdim",
    "Kendime güvenim her geçen gün artıyor",
    "Yoğun tempoda çalışmak bazen zorlayıcı oluyor",
    "Bugün beklediğim cevabı nihayet aldım",
    "Bu konuyu öğrenmek sandığımdan daha keyifliydi",
    "Zor bir günün ardından biraz dinlenmek istiyorum",
    "Yeni hedefler koymak bana yön veriyor",
    "Bugün işler beklediğimden daha hızlı ilerledi",
    "Kendi gelişimimi net bir şekilde fark edebiliyorum",
    "Bu fikrin işe yarayacağını düşünüyorum",
    "Bugün oldukça odaklı çalıştım",
    "Bazen her şeyi akışına bırakmak gerekiyor",
    "Uzun zamandır böyle üretken hissetmemiştim",
    "Bu sorunun çözümü düşündüğümden daha basitti",
    "Yeni alışkanlıklar edinmeye çalışıyorum",
    "Bugün küçük ama önemli bir adım attım",
    "Bu ortamda çalışmak beni motive ediyor",
    "Kendimi daha iyi tanımaya başladım",
    "Bugün aldığım geri bildirimler çok faydalıydı",
    "Zamanla her şeyin daha iyi olacağına inanıyorum",
    "Bu süreç bana sabırlı olmayı öğretiyor",
    "Bugün biraz dalgın hissediyorum",
    "Uzun süredir üzerinde düşündüğüm kararı verdim",
    "Yeni bir bakış açısı kazanmak bana iyi geldi",
    "Bugün beklenmedik bir sorunla karşılaştım",
    "Bu işi daha iyi yapabileceğimi biliyorum",
    "Kendi sınırlarımı zorlamaktan çekinmiyorum",
    "Bugün motivasyonum dalgalıydı",
    "Bu deneyim bana özgüven kazandırdı",
    "Yeni şeyler denemek beni geliştiriyor",
    "Bugün hedeflerime biraz daha yaklaştım",
    "Bazen durup düşünmek gerekiyor",
    "Bu alanda kendimi geliştirmek istiyorum",
    "Bugün işler biraz karmaşıktı",
    "Zorlandığım konular üzerine daha çok çalışmalıyım",
    "Bu başarı beni daha da motive etti",
    "Bugün sakin ve dengeli hissettim",
    "Uzun vadeli düşünmenin önemini fark ettim",
    "Bu sürecin sonunda iyi sonuçlar alacağıma inanıyorum",
    "Bugün kendime karşı daha anlayışlıydım",
    "Yeni bir plan yapmak beni rahatlattı",
    "Bugün oldukça disiplinli davrandım",
    "Bu karar hayatımda önemli bir dönüm noktası olabilir",
    "Kendimi geliştirmek için doğru yolda olduğumu hissediyorum",
    "Bugün üretken bir şekilde geçti"
]

#tokenization and creating n-gram sequences
tokenizer=Tokenizer()
tokenizer.fit_on_texts(data)
total_words=len(tokenizer.word_index)+1
print("Total words:", total_words)

input_sequences = []
for text in data:
    tokenlist=tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(tokenlist)):
        n_gram_sequence = tokenlist[:i+1]
        input_sequences.append(n_gram_sequence)

print(f"Total input sequences: {(input_sequences)}")

#padding

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

print(f"Padded input sequences: {input_sequences}")


X=input_sequences[:,:-1] #n-1 kelimeyi girdi olarak al
y=input_sequences[:,-1]  #son kelimeyi çıktı olarak al

y=tensorflow.keras.utils.to_categorical(y, num_classes=total_words)

print(f"target: {y}")

model=Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy']    )
#model katman özeti
model.summary()

model.fit(X, y, epochs=100, verbose=1) 

#model test

def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted, axis=1)[0]
        
        predicted_word = tokenizer.index_word[predicted_index]
        seed_text= seed_text+" "+predicted_word 

    return seed_text

print(generate_text("Bugün", 4))

model.save("lstm_text_generation_model.h5")
print("Model kaydedildi.")