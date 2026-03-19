import json
import random

# --- EINSTELLUNGEN ---
INPUT_FILE = "jack_massive_8b_dataset.json"          # Deine generierte Datei
OUTPUT_FILE = "jack_massive_8b_dataset_shuffled.json" # Die perfekt gemischte End-Datei

def shuffle_dataset():
    print(f"Lese Datei '{INPUT_FILE}' ein...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"✅ {len(data)} Einträge gefunden. Mische kräftig durch...")
        
        # Hier passiert die Magie: Die Reihenfolge wird komplett zufällig!
        random.shuffle(data)
        
        print("Speichere die gemischte Datei...")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        print(f"🎉 FERTIG! Deine perfekten, gemischten Trainingsdaten liegen jetzt in: {OUTPUT_FILE}")
        print("Du kannst diese Datei jetzt direkt für dein Llama 8B Training benutzen! 🚀")
        
    except FileNotFoundError:
        print(f"❌ FEHLER: Konnte die Datei '{INPUT_FILE}' nicht finden. Ist das Generierungs-Skript schon fertig?")
    except Exception as e:
        print(f"❌ Ein unerwarteter Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    shuffle_dataset()