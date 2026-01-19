# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Gerekli kütüphaneleri içe aktarma
# pip install openai python-dotenv scipy sounddevice numpy
from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import os
import uuid
import re
from datetime import datetime
from dotenv import load_dotenv
import logging
import numpy as np
import time

# Şu anki tarih ve saat ile log dosyası adı oluşturma
now = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/konusma_{now}.log"

# logs klasörünü oluşturma (yoksa)
os.makedirs("logs", exist_ok=True)

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# .env dosyasından çevre değişkenlerini yükleme
load_dotenv()

# OpenAI istemcisini başlatma
client = OpenAI()

# Sabitler
FS = 44100  # Örnekleme frekansı (Hz)
CHUNK_DURATION = 0.1  # Her chunk 100ms (0.1 saniye)
CHUNK_SIZE = int(FS * CHUNK_DURATION)

# Ses aktivasyon parametreleri
SILENCE_THRESHOLD = 0.02  # Sessizlik eşiği (deneyerek ayarlayın: 0.01-0.05 arası)
SPEECH_THRESHOLD = 0.03   # Konuşma başlangıç eşiği
MIN_SPEECH_DURATION = 0.5  # Minimum konuşma süresi (saniye)
SILENCE_DURATION = 1.5     # Konuşma bittikten sonra kaç saniye sessizlik beklenecek
MAX_RECORDING_DURATION = 15  # Maksimum kayıt süresi (saniye)

BANNED_WORDS = ["salak", "mal", "aptal"]

def filter_bad_words(text):
    """Metindeki yasaklı kelimeleri tespit edip '*' ile değiştirir"""
    filtered_text = text
    for word in BANNED_WORDS:
        if re.search(rf"\b{word}\b", text, flags=re.IGNORECASE):
            logger.warning(f"Zararlı kelime bulundu: {word}")
        filtered_text = re.sub(rf"\b{word}\b", "****", filtered_text, flags=re.IGNORECASE)
    return filtered_text

def calculate_rms(audio_chunk):
    """
    Ses parçasının RMS (Root Mean Square) değerini hesaplar
    RMS, ses seviyesini ölçmek için kullanılır
    """
    return np.sqrt(np.mean(audio_chunk**2))

def record_with_vad():
    """
    Ses Aktivasyon Tespitiyle (VAD) kayıt yapar
    Sessizlikte bekler, konuşma tespit edince kaydeder, 
    konuşma bitince otomatik durur
    
    Returns:
        Kaydedilen ses verisi (numpy array) veya None
    """
    print("\n" + "="*60)
    print("🎧 DİNLİYORUM... (Konuşmaya başlayın)")
    print("="*60)
    
    audio_buffer = []  # Kaydedilen ses parçaları
    is_speaking = False  # Şu an konuşuyor mu?
    silence_chunks = 0  # Art arda kaç sessiz chunk geldi
    speech_chunks = 0   # Art arda kaç konuşma chunk'ı geldi
    total_chunks = 0    # Toplam kayıt chunk sayısı
    
    # Maksimum chunk sayısını hesapla
    max_chunks = int(MAX_RECORDING_DURATION / CHUNK_DURATION)
    silence_chunks_needed = int(SILENCE_DURATION / CHUNK_DURATION)
    speech_chunks_needed = int(MIN_SPEECH_DURATION / CHUNK_DURATION)
    
    # Sürekli dinleme döngüsü
    with sd.InputStream(samplerate=FS, channels=1, blocksize=CHUNK_SIZE) as stream:
        while True:
            # Mikrofondan bir chunk (parça) oku
            audio_chunk, _ = stream.read(CHUNK_SIZE)
            rms = calculate_rms(audio_chunk)
            
            # Konuşma henüz başlamadıysa
            if not is_speaking:
                # Ses seviyesi eşiği aştı mı?
                if rms > SPEECH_THRESHOLD:
                    speech_chunks += 1
                    # Yeterince uzun konuşma tespit edildi mi?
                    if speech_chunks >= speech_chunks_needed:
                        is_speaking = True
                        silence_chunks = 0
                        print("🔴 KAYIT BAŞLADI - Konuşun...")
                        logger.info(f"Konuşma başladı (RMS: {rms:.4f})")
                        # Önceki chunk'ları da dahil et (konuşmanın başını kaçırmamak için)
                        audio_buffer.append(audio_chunk)
                else:
                    speech_chunks = 0  # Sessizlik, sayacı sıfırla
                    
            # Konuşma başladıysa
            else:
                audio_buffer.append(audio_chunk)
                total_chunks += 1
                
                # Sessizlik kontrolü
                if rms < SILENCE_THRESHOLD:
                    silence_chunks += 1
                    # Görsel geri bildirim
                    if silence_chunks % 5 == 0:
                        print(".", end="", flush=True)
                else:
                    silence_chunks = 0  # Tekrar konuşma başladı, sayacı sıfırla
                
                # Konuşma bitti mi? (yeterince uzun sessizlik)
                if silence_chunks >= silence_chunks_needed:
                    print("\n✓ KAYIT BİTTİ")
                    logger.info(f"Konuşma bitti ({total_chunks * CHUNK_DURATION:.1f} saniye)")
                    break
                
                # Maksimum süre aşıldı mı?
                if total_chunks >= max_chunks:
                    print("\n⏱️ Maksimum süre doldu")
                    logger.warning("Maksimum kayıt süresi aşıldı")
                    break
    
    # Kaydedilen ses parçalarını birleştir
    if audio_buffer:
        return np.concatenate(audio_buffer, axis=0)
    return None

def save_audio(audio_data, filename):
    """
    Ses verisini WAV dosyası olarak kaydeder
    
    Args:
        audio_data: Numpy array ses verisi
        filename: Kaydedilecek dosya adı
    """
    write(filename, FS, audio_data)
    logger.info(f"Ses dosyası kaydedildi: {filename}")

def transcribe_with_whisper(audio_path):
    """OpenAI Whisper API kullanarak ses dosyasını metne çevirir"""
    logger.info("Whisper ile ses yazıya çevriliyor...")
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="tr",
            prompt="Bu bir Türkçe konuşmadır."
        )       
    return transcript.text

def get_gpt_response(messages):
    """GPT modeline mesaj geçmişini gönderip yanıt alır"""
    logger.info("GPT yanıt üretiyor...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content

def speak_response(text):
    """
    GPT yanıtını ekrana yazdırır ve sesli okuma için hazırlar
    (İsteğe bağlı: TTS eklenebilir)
    """
    print("\n" + "="*60)
    print("🤖 ASİSTAN:")
    print("-"*60)
    print(text)
    print("="*60 + "\n")

if __name__ == "__main__":
    logger.info("----------- SES AKTİVASYONLU GPT ASİSTAN BAŞLADI -----------")
    logger.info(f"Konuşma log dosyası: {log_file}")
    logger.info(f"Ses eşiği: {SPEECH_THRESHOLD}, Sessizlik eşiği: {SILENCE_THRESHOLD}")
    
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║          SES AKTİVASYONLU GPT ASİSTAN                      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\n💡 İPUCU: 'çık', 'kapat' veya 'bitir' diyerek çıkabilirsiniz\n")

    # Sistem mesajı ile konuşma geçmişini başlatma
    messages = [{
        "role": "system", 
        "content": "Sen yardımsever, doğal ve samimi bir sesli asistansın. Kısa ve öz cevaplar ver."
    }]

    # Ana döngü - sürekli dinleme
    conversation_count = 0
    
    try:
        while True:
            conversation_count += 1
            logger.info(f"--- Konuşma #{conversation_count} bekleniyor ---")
            
            # Ses aktivasyonuyla kayıt yap
            audio_data = record_with_vad()
            
            if audio_data is None:
                logger.warning("Ses kaydı alınamadı, tekrar deneniyor...")
                continue
            
            # Benzersiz dosya adı oluştur ve kaydet
            uid = str(uuid.uuid4())
            audio_file = f"record_{uid}.wav"
            save_audio(audio_data, audio_file)
            
            # Ses kaydını metne çevir
            try:
                question = transcribe_with_whisper(audio_file)
                logger.info(f"Kullanıcı: {question}")
                
                # Boş veya çok kısa transkript kontrolü
                if not question or len(question.strip()) < 2:
                    logger.warning("Transkript çok kısa veya boş, atlanıyor...")
                    os.remove(audio_file)
                    continue
                
                # Kötü kelimeleri filtrele
                filtered_question = filter_bad_words(question)
                
                if filtered_question != question:
                    logger.info(f"Kullanıcı (filtreli): {filtered_question}")
                
                # Çıkış komutları kontrolü
                exit_keywords = ["çık", "kapat", "bitir", "hoşça kal", "görüşürüz"]
                if any(keyword in filtered_question.lower() for keyword in exit_keywords):
                    logger.info("Çıkış komutu algılandı")
                    print("\n👋 Görüşmek üzere! Hoşça kalın.\n")
                    break
                
                # Kullanıcı mesajını geçmişe ekle
                messages.append({"role": "user", "content": filtered_question})
                
                # GPT'den yanıt al
                answer = get_gpt_response(messages)
                logger.info(f"GPT: {answer}")
                
                # GPT yanıtını geçmişe ekle
                messages.append({"role": "assistant", "content": answer})
                
                # Yanıtı ekrana yazdır
                speak_response(answer)
                
                # Geçici ses dosyasını sil
                os.remove(audio_file)
                logger.info(f"Ses dosyası silindi: {audio_file}")
                
                # Kısa bekleme (kullanıcının yanıtı okuması için)
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Hata oluştu: {e}")
                print(f"❌ Bir hata oluştu: {e}")
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                continue
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Program kullanıcı tarafından durduruldu")
        logger.info("Program Ctrl+C ile durduruldu")
    
    logger.info("----------- SES AKTİVASYONLU GPT ASİSTAN SONLANDI -----------")
    print("\n✓ Program sonlandı.\n")