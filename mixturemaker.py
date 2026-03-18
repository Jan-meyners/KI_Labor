import json
import random

# EINSTELLUNGEN
INPUT_FILE = "jack_datenbank.json"
OUTPUT_FILE = "jack_final_mixture.jsonl" # Profi-Tipp: .jsonl ist besser für Unsloth
ANZAHL_EXPERTEN_FRAGEN = 333

def upgrade_to_expert_style(antwort):
    expert_prompt = (
        f"Zuerst analysiere ich die aktuelle Faktenlage der JIM-Studie zu dieser Frage. "
        f"Dann betrachte ich die pädagogische Komponente für die Altersgruppe. "
        f"Basierend darauf lautet meine Empfehlung: {antwort}"
    )
    return expert_prompt

def create_mixture():
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"Datensatz geladen: {len(data)} Einträge.")
        
        # KEY-CHECK: Wir finden heraus, wie deine Spalten heißen
        first_entry = data[0]
        # Wir suchen nach den Schlüsseln für Frage und Antwort
        q_key = next((k for k in ["frage", "question", "instruction", "input"] if k in first_entry), None)
        a_key = next((k for k in ["antwort", "answer", "output", "response"] if k in first_entry), None)

        if not q_key or not a_key:
            print(f"❌ Fehler: Konnte keine Frage- oder Antwort-Spalten finden!")
            print(f"Vorhandene Schlüssel sind: {list(first_entry.keys())}")
            return

        print(f"Verwende '{q_key}' als Frage und '{a_key}' als Antwort.")
        
        random.shuffle(data)
        
        final_data = []
        for i, entry in enumerate(data):
            frage = entry[q_key]
            original_antwort = entry[a_key]
            
            if i < ANZAHL_EXPERTEN_FRAGEN:
                final_antwort = upgrade_to_expert_style(original_antwort)
            else:
                final_antwort = original_antwort
            
            # Einheitliches Format für das Training
            jsonl_line = {
                "instruction": "Du bist Professor Jack, ein Experten-KI für Medienpädagogik. Antworte präzise und beratend.",
                "input": frage,
                "output": final_antwort
            }
            final_data.append(jsonl_line)
            
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for line in final_data:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                
        print(f"--- FERTIG ---")
        print(f"Datei erstellt: {OUTPUT_FILE}")
        print(f"Experten-Style: {min(ANZAHL_EXPERTEN_FRAGEN, len(data))} | Normal-Style: {max(0, len(data)-ANZAHL_EXPERTEN_FRAGEN)}")

    except Exception as e:
        print(f"Fehler: {e}")

if __name__ == "__main__":
    create_mixture()