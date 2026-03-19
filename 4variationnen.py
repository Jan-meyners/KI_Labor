import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- EINSTELLUNGEN ---
INPUT_FILE = "jack_datenbank.json"  # Trag hier deine aktuelle Datei ein (egal ob .json oder .jsonl)
OUTPUT_FILE = "jack_massive_8b_dataset.json" # 👈 HIER: Jetzt eine echte .json Datei

print("🤖 Lade Llama 3.2 3B als Hilfsarbeiter zum Umschreiben der Fragen...")

# Wir laden das Basis-Modell in 4-Bit
model_name = "unsloth/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quant_config, device_map="auto"
)

def generate_variations(original_question):
    prompt = f"""Schreibe genau 4 verschiedene, natürlich klingende Variationen für diese Frage. 
Behalte den genauen Sinn bei. Nutze verschiedene Satzbauten.
Schreibe KEINEN Einleitungstext, nur die 4 Fragen, jede in einer neuen Zeile, beginnend mit einem Strich (-).

Originale Frage: {original_question}

Antwort:
-"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.7, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        response_part = text.split("Antwort:\n-")[1]
        lines = response_part.split('\n')
        variations = []
        for line in lines:
            clean_line = line.strip().lstrip('-').strip()
            clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
            if clean_line and len(clean_line) > 5:
                variations.append(clean_line)
        return variations[:4]
    except:
        return []

# --- HAUPTPROZESS ---
# --- HAUPTPROZESS ---
def main():
    print("Lese Daten ein...")
    data = []
    
    # Der "kugelsichere" Lader
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data.append(json.loads(line))
                except:
                    pass

    if len(data) == 0:
        print("❌ FEHLER: Die Datei ist komplett leer oder konnte nicht gelesen werden.")
        return

    print(f"✅ Erfolgreich {len(data)} echte Fragen gefunden.")
    print("Starte Generierung... (Das dauert jetzt ein bisschen!)\n")
    
    augmented_data = []
    
    for i, entry in enumerate(data):
        # ANPASSUNG AN DEIN BILD: 
        # Deine Frage steht in "instruction", die Antwort in "output"
        original_q = entry.get("instruction", "")
        answer = entry.get("output", "")
        
        # Wir fügen für das neue Training den System-Prompt hinzu, 
        # und speichern die Frage als "input", damit Unsloth glücklich ist!
        system_instruction = "Du bist Professor Jack, ein Experten-KI für Medienpädagogik. Antworte präzise und beratend."
        
        if not original_q or not answer:
            continue
            
        # 1. Original speichern (im perfekten Format)
        augmented_data.append({"instruction": system_instruction, "input": original_q, "output": answer})
        
        # 2. Variationen generieren
        print(f"Bearbeite Frage {i+1}/{len(data)}: {original_q[:50]}...")
        new_questions = generate_variations(original_q)
        
        # 3. Variationen speichern
        for nq in new_questions:
            augmented_data.append({"instruction": system_instruction, "input": nq, "output": answer})

    # Finales Speichern als echtes JSON-Array mit schöner Einrückung
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=4)
            
    print(f"\n🎉 FERTIG! Alter Datensatz: {len(data)} -> Neuer Datensatz: {len(augmented_data)}")
    print(f"Gespeichert in {OUTPUT_FILE}. Du hast jetzt eine echte, sauber formatierte .json Datei!")

if __name__ == "__main__":
    main()