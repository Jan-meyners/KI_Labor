import torch
import json
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

print("Lade Werkzeuge für das KI-Training... Bitte warten!")

# 1. Das "Schulbuch" (Unsere frisch generierten Trainingsdaten laden)
daten_pfad = "jack_datenbank.json"
if not os.path.exists(daten_pfad):
    print(f"FEHLER: Die Datei '{daten_pfad}' wurde nicht gefunden!")
    exit()

with open(daten_pfad, "r", encoding="utf-8") as f:
    daten = json.load(f)
mein_schulbuch = Dataset.from_list(daten)
print(f"Erfolgreich {len(daten)} Fragen aus der Datenbank geladen.")

# 2. Der "Schrumpfstrahler" (4-Bit Quantisierung für 8GB VRAM)
# Für deine RTX 4060 Ti nutzen wir jetzt das superschnelle bfloat16 (bf16) Format!
schrumpf_einstellungen = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # <-- Der Turbo-Boost für RTX 4000er!
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 3. Das Basis-Modell (Llama 3.2 - 3B Instruct) und den Tokenizer laden
modell_name = "unsloth/Llama-3.2-3B-Instruct" 

tokenizer = AutoTokenizer.from_pretrained(modell_name)
# Wichtig für Llama 3: Ein "Füllzeichen" (Pad-Token) definieren, falls Sätze unterschiedlich lang sind
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Lade das Basis-Modell herunter (das kann beim ersten Mal ein paar Minuten dauern)...")
modell = AutoModelForCausalLM.from_pretrained(
    modell_name,
    quantization_config=schrumpf_einstellungen,
    device_map="auto" # Erkennt und nutzt automatisch deine RTX 4060 Ti
)

# 4. Das "Effektgerät" (LoRA) einstellen - Maximales Verständnis!
lora_einstellungen = LoraConfig(
    r=16, 
    lora_alpha=32,
    # Wir trainieren jetzt ALLE wichtigen Schichten im Gehirn, damit Jack Zusammenhänge tiefgreifend versteht:
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    task_type="CAUSAL_LM",
    bias="none"
)

# 5. Wie sollen die Daten aussehen? (Klare Struktur für Jack)
def text_formatieren(beispiel):
    # Das Format trennt Frage und Antwort sauber und gibt eine klare Struktur vor
    return f"### Frage:\n{beispiel['instruction']}\n\n### Antwort:\n{beispiel['output']}"

# 6. Die Trainingsregeln (Perfekt auf 8 Epochen und deine RTX 4060 Ti abgestimmt)
trainings_regeln = TrainingArguments(
    output_dir="./professor_jack_checkpoints", # Hier landen die Zwischenspeicherungen
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4, # Schont den 8GB VRAM, simuliert aber große Datenblöcke
    learning_rate=2e-5,            # Sanfte Lernrate, damit er nicht zum "Papagei" wird
    num_train_epochs=8,            # 8 Durchläufe (Epochen) für festes Wissen
    logging_steps=5,               # Zeigt dir alle 5 Schritte den Fortschritt im Terminal
    bf16=True,                     # <-- TURBO für RTX 4060 Ti aktiviert!
    fp16=False,                    
    gradient_checkpointing=True,   # Verhindert, dass die Grafikkarte wegen Speichermangel abstürzt
    save_strategy="epoch",         # <-- UNSERE LEBENSVERSICHERUNG: Speichert nach jeder der 8 Epochen eine Kopie!
    optim="paged_adamw_8bit"       # Ein spezieller Speicher-Optimierer
)

# Alles zusammenbauen
trainer = SFTTrainer(
    model=modell,
    train_dataset=mein_schulbuch,
    peft_config=lora_einstellungen,
    formatting_func=text_formatieren,
    args=trainings_regeln,
)

# 7. Startschuss!
print("\n--- Mache die Leinen los... Das Training auf deiner RTX 4060 Ti beginnt! ---")
trainer.train()

# 8. Das absolut finale Modell speichern
endgueltiger_pfad = "./professor_jack_fertig"
trainer.model.save_pretrained(endgueltiger_pfad)
tokenizer.save_pretrained(endgueltiger_pfad)
print(f"\n🎉 TRAINING BEENDET! 🎉")
print(f"Dein fertiger 'Professor Jack' wurde im Ordner '{endgueltiger_pfad}' gespeichert.")
print("Hinweis: Im Ordner 'professor_jack_checkpoints' findest du die Versionen der einzelnen Epochen, falls Epoche 8 zu 'auswendig gelernt' wirkt!")