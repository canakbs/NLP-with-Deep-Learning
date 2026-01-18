import os
from datetime import datetime
from asistant import get_gemini_response
from database import (
    initialize_db, add_event, add_note, 
    get_events, get_notes, delete_event, delete_note
)

class PersonalOrganizer:
    def __init__(self):
        initialize_db()
        self.commands = {
            "1": ("Not Ekle", self.add_note_interactive),
            "2": ("Etkinlik Ekle", self.add_event_interactive),
            "3": ("Notları Göster", self.show_notes),
            "4": ("Etkinlikleri Göster", self.show_events),
            "5": ("Not Sil", self.delete_note_interactive),
            "6": ("Etkinlik Sil", self.delete_event_interactive),
            "7": ("AI Asistan ile Sohbet", self.chat_with_ai),
            "8": ("Çıkış", self.exit_app)
        }
        
    def clear_screen(self):
        """Ekranı temizler"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Başlık gösterir"""
        print("\n" + "="*60)
        print("🗂️  KİŞİSEL ORGANİZER - AI DESTEKLİ ASISTAN")
        print("="*60 + "\n")
    
    def print_menu(self):
        """Menüyü gösterir"""
        print("📋 MENÜ:")
        print("-" * 60)
        for key, (description, _) in self.commands.items():
            print(f"  [{key}] {description}")
        print("-" * 60)
    
    def add_note_interactive(self):
        """Not ekleme işlemi"""
        print("\n📝 NOT EKLEME")
        print("-" * 60)
        note = input("Notunuzu girin (iptal için 'q'): ").strip()
        
        if note.lower() == 'q':
            print("❌ İşlem iptal edildi.")
            return
        
        if not note:
            print("⚠️  Not boş olamaz!")
            return
        
        add_note(note)
        print("✅ Not başarıyla eklendi!")
    
    def add_event_interactive(self):
        """Etkinlik ekleme işlemi"""
        print("\n📅 ETKİNLİK EKLEME")
        print("-" * 60)
        event = input("Etkinlik adı (iptal için 'q'): ").strip()
        
        if event.lower() == 'q':
            print("❌ İşlem iptal edildi.")
            return
        
        if not event:
            print("⚠️  Etkinlik adı boş olamaz!")
            return
        
        while True:
            event_date = input("Tarih (YYYY-MM-DD formatında): ").strip()
            
            if event_date.lower() == 'q':
                print("❌ İşlem iptal edildi.")
                return
            
            try:
                # Tarih formatını kontrol et
                datetime.strptime(event_date, '%Y-%m-%d')
                break
            except ValueError:
                print("⚠️  Geçersiz tarih formatı! Lütfen YYYY-MM-DD formatında girin.")
        
        add_event(event, event_date)
        print("✅ Etkinlik başarıyla eklendi!")
    
    def show_notes(self):
        """Notları gösterir - Modüler yapı"""
        print("\n📝 NOTLARIM")
        print("=" * 60)
        notes = get_notes()
        
        if not notes:
            print("📭 Henüz not yok.")
            return
        
        for idx, note in enumerate(notes, 1):
            # get_notes() -> (content, timestamp)
            content = note[0] if len(note) > 0 else "Boş not"
            timestamp = note[1] if len(note) > 1 else "Tarih bilinmiyor"
            
            print(f"\n[{idx}] {content}")
            print(f"    🕐 {timestamp}")
        print("-" * 60)
    
    def show_events(self):
        """Etkinlikleri gösterir - Modüler yapı"""
        print("\n📅 ETKİNLİKLERİM")
        print("=" * 60)
        events = get_events()
        
        if not events:
            print("📭 Henüz etkinlik yok.")
            return
        
        for idx, event in enumerate(events, 1):
            # get_events() -> (event, event_date)
            event_name = event[0] if len(event) > 0 else "Belirsiz etkinlik"
            event_date = event[1] if len(event) > 1 else "Tarih bilinmiyor"
            
            print(f"\n[{idx}] {event_name}")
            print(f"    📆 Tarih: {event_date}")
        print("-" * 60)
    
    def delete_note_interactive(self):
        """Not silme işlemi - Modüler yapı"""
        print("\n🗑️  NOT SİLME")
        print("-" * 60)
        
        notes = get_notes()
        if not notes:
            print("📭 Silinecek not yok.")
            return
        
        # Notları listele
        for idx, note in enumerate(notes, 1):
            content = note[0] if len(note) > 0 else "Boş not"
            # İlk 50 karakteri göster
            display_text = content[:50] + "..." if len(content) > 50 else content
            timestamp = note[1] if len(note) > 1 else ""
            print(f"[{idx}] {display_text} ({timestamp})")
        
        try:
            choice = input("\nSilmek istediğiniz notun numarasını girin (iptal için 'q'): ").strip()
            
            if choice.lower() == 'q':
                print("❌ İşlem iptal edildi.")
                return
            
            choice = int(choice)
            if 1 <= choice <= len(notes):
                delete_note(choice)
                print("✅ Not başarıyla silindi!")
            else:
                print("⚠️  Geçersiz numara!")
        except ValueError:
            print("⚠️  Lütfen geçerli bir numara girin!")
        except Exception as e:
            print(f"⚠️  Hata oluştu: {e}")
    
    def delete_event_interactive(self):
        """Etkinlik silme işlemi - Modüler yapı"""
        print("\n🗑️  ETKİNLİK SİLME")
        print("-" * 60)
        
        events = get_events()
        if not events:
            print("📭 Silinecek etkinlik yok.")
            return
        
        # Etkinlikleri listele
        for idx, event in enumerate(events, 1):
            event_name = event[0] if len(event) > 0 else "Belirsiz etkinlik"
            event_date = event[1] if len(event) > 1 else "Tarih bilinmiyor"
            print(f"[{idx}] {event_name} - {event_date}")
        
        try:
            choice = input("\nSilmek istediğiniz etkinliğin numarasını girin (iptal için 'q'): ").strip()
            
            if choice.lower() == 'q':
                print("❌ İşlem iptal edildi.")
                return
            
            choice = int(choice)
            if 1 <= choice <= len(events):
                delete_event(choice)
                print("✅ Etkinlik başarıyla silindi!")
            else:
                print("⚠️  Geçersiz numara!")
        except ValueError:
            print("⚠️  Lütfen geçerli bir numara girin!")
        except Exception as e:
            print(f"⚠️  Hata oluştu: {e}")
    
    def format_notes_for_context(self):
        """Notları AI için formatlar"""
        notes = get_notes()
        if not notes:
            return "Henüz not yok."
        
        formatted = []
        for idx, note in enumerate(notes, 1):
            content = note[0] if len(note) > 0 else "Boş not"
            timestamp = note[1] if len(note) > 1 else "Tarih bilinmiyor"
            formatted.append(f"{idx}. {content} (Tarih: {timestamp})")
        
        return "\n".join(formatted)
    
    def format_events_for_context(self):
        """Etkinlikleri AI için formatlar"""
        events = get_events()
        if not events:
            return "Henüz etkinlik yok."
        
        formatted = []
        for idx, event in enumerate(events, 1):
            event_name = event[0] if len(event) > 0 else "Belirsiz etkinlik"
            event_date = event[1] if len(event) > 1 else "Tarih bilinmiyor"
            formatted.append(f"{idx}. {event_name} - {event_date}")
        
        return "\n".join(formatted)
    
    def chat_with_ai(self):
        """AI asistan ile sohbet - Modüler yapı"""
        print("\n🤖 AI ASISTAN İLE SOHBET")
        print("=" * 60)
        print("💡 Not ve etkinlikleriniz hakkında soru sorabilirsiniz.")
        print("   Çıkmak için 'çıkış', 'exit' veya 'q' yazın.\n")
        
        # Mevcut not ve etkinlikleri formatla
        notes_text = self.format_notes_for_context()
        events_text = self.format_events_for_context()
        
        context = f"""
Sen bir kişisel asistansın. Kullanıcının not ve etkinliklerini kullanarak sorularına yanıt ver.

NOTLAR:
{notes_text}

ETKİNLİKLER:
{events_text}

Kullanıcıya yardımcı ol, hatırlatıcılar ver ve organizasyonunda destek ol.
Türkçe yanıt ver ve samimi bir ton kullan.
"""
        
        while True:
            user_message = input("\n👤 Siz: ").strip()
            
            if not user_message:
                continue
            
            if user_message.lower() in ["çıkış", "exit", "quit", "q"]:
                print("👋 Sohbetten çıkılıyor...\n")
                break
            
            try:
                full_prompt = f"{context}\n\nKullanıcı: {user_message}\nAsistan:"
                response = get_gemini_response(full_prompt)
                print(f"\n🤖 Asistan: {response}")
            except Exception as e:
                print(f"⚠️  Hata oluştu: {e}")
                print("💡 Lütfen internet bağlantınızı ve API anahtarınızı kontrol edin.")
    
    def exit_app(self):
        """Uygulamadan çıkış"""
        print("\n👋 Görüşmek üzere! İyi günler dileriz.")
        exit(0)
    
    def run(self):
        """Ana döngü"""
        self.clear_screen()
        self.print_header()
        print("🎉 Hoş geldiniz! Kişisel organizör asistanınız hazır.\n")
        
        while True:
            try:
                self.print_menu()
                choice = input("\n🔹 Seçiminiz: ").strip()
                
                if choice in self.commands:
                    print()
                    _, action = self.commands[choice]
                    action()
                else:
                    print("\n⚠️  Geçersiz seçim! Lütfen menüden bir numara seçin.")
                
                input("\n⏎ Devam etmek için Enter'a basın...")
                self.clear_screen()
                self.print_header()
            except KeyboardInterrupt:
                print("\n\n👋 Program sonlandırılıyor...")
                break
            except Exception as e:
                print(f"\n⚠️  Beklenmeyen bir hata oluştu: {e}")
                input("\n⏎ Devam etmek için Enter'a basın...")


if __name__ == "__main__":
    try:
        organizer = PersonalOrganizer()
        organizer.run()
    except Exception as e:
        print(f"⚠️  Program başlatılamadı: {e}")