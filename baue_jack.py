import ollama
import json
import re

# --- EINSTELLUNGEN ---
INPUT_DATEI = "rohdaten.txt"
OUTPUT_DATEI = "jack_datenbank.json"
ZIEL_ANZAHL = 2000
CHUNK_GROESSE = 3000  
MODELL = "llama3.2:3b"

def ist_schrott(text_chunk):
    """Prüft, ob der Textblock offensichtlich nur aus Impressum oder Menüs besteht."""
    # Müll-Wörter, die auf Impressum oder Navigation hindeuten
    muell_woerter = ["impressum", "datenschutzerklärung", "umsatzsteuer-id", "haftungsausschluss", "alle rechte vorbehalten", "cookie-einstellungen", "agb"]
    
    treffer = 0
    text_lower = text_chunk.lower()
    for wort in muell_woerter:
        if wort in text_lower:
            treffer += 1
            
    # Wenn 2 oder mehr dieser Wörter im Block sind, werfen wir ihn weg
    if treffer >= 2:
        return True
    return False

def generiere_fragen(text_chunk):
    prompt = f"""
    Du bist ein Daten-Generator für ein KI-Training.
    Lies den folgenden Textausschnitt und erstelle daraus bis zu 5 hochwertige Frage-Antwort-Paare.
    
    EXTREM WICHTIG - SCHROTT-FILTER:
    Wenn der Textausschnitt kein nützliches Wissen über Medienpädagogik oder Studien (wie JIM/KIM) enthält, gib einfach [] zurück!
    
    REGELN:
    1. JEDE Frage MUSS zwingend mit "Jack, " beginnen (z.B. "Jack, wie alt...").
    2. Die Antworten kommen von "Jack", einem professionellen, empathischen Medien-Experten.
    3. EXTREM WICHTIG: Antworte IMMER in vollständigen, ausführlichen und freundlichen Sätzen! Erkläre den Kontext. KEIN Telegramm-Stil, keine Stichpunkte!
    4. Gib AUSSCHLIESSLICH ein valides JSON-Array zurück.
    
    FORMAT-BEISPIEL:
    [
      {{"instruction": "Jack, wie alt sind die in der aktuellen JIM-Studie befragten Jugendlichen?", "output": "Hallo! In der aktuellen JIM-Studie werden traditionell Jugendliche im Alter von zwölf bis 19 Jahren befragt, um ein genaues Bild dieser Altersgruppe zu erhalten."}}
    ]
    
    TEXTAUSSCHNITT:
    {text_chunk}
    """
    
    try:
        response = ollama.chat(model=MODELL, messages=[{'role': 'user', 'content': prompt}])
        antwort_text = response['message']['content']
        
        match = re.search(r'\[.*\]', antwort_text, re.DOTALL)
        if match:
            json_daten = json.loads(match.group(0))
            return json_daten
        else:
            return []
    except Exception as e:
        # Falls die KI kein JSON liefert (oft passiert das bei Schrott-Texten), ignorieren wir es
        return []

def main():
    print("Lese Rohdaten ein...")
    with open(INPUT_DATEI, "r", encoding="utf-8") as f:
        kompletter_text = f.read()

    chunks = [kompletter_text[i:i+CHUNK_GROESSE] for i in range(0, len(kompletter_text), CHUNK_GROESSE)]
    print(f"Text in {len(chunks)} Häppchen aufgeteilt.")

    alle_fragen = []
    
    for i, chunk in enumerate(chunks):
        if len(alle_fragen) >= ZIEL_ANZAHL:
            break
            
        # 1. Stufe: Den Python-Türsteher fragen
        if ist_schrott(chunk):
            print(f"Häppchen {i+1} übersprungen (Wahrscheinlich Impressum/Schrott).")
            continue
            
        print(f"Verarbeite Häppchen {i+1}/{len(chunks)}... (Aktuell: {len(alle_fragen)} Fragen gesammelt)")
        
        # 2. Stufe: Die KI fragen
        neue_fragen = generiere_fragen(chunk)
        
        for frage in neue_fragen:
            if "instruction" in frage and "output" in frage:
                alle_fragen.append(frage)
                
        with open(OUTPUT_DATEI, "w", encoding="utf-8") as f:
            json.dump(alle_fragen[:ZIEL_ANZAHL], f, ensure_ascii=False, indent=2)

    print(f"\nFERTIG! {len(alle_fragen[:ZIEL_ANZAHL])} saubere Fragen gespeichert.")

if __name__ == "__main__":
    main()