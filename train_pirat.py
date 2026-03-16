import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

print("Lade Werkzeuge... Bitte warten!")

# 1. Das "Schulbuch" (Unsere Trainingsdaten)
# Je mehr solcher Paare du später hast, desto schlauer wird dein Modell!
with open("datenbank.json", "r", encoding="utf-8") as f:
    daten = json.load(f)
mein_schulbuch = Dataset.from_list(daten)

# 2. Der "Schrumpfstrahler" (4-Bit Quantisierung für deine 8GB Grafikkarte)
# Ohne das würde das Modell nicht auf deine Karte passen.
schrumpf_einstellungen = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 3. Das Basis-Modell (Qwen) und die Lego-Stein-Maschine (Tokenizer) laden
modell_name = "Qwen/Qwen2.5-1.5B" 

tokenizer = AutoTokenizer.from_pretrained(modell_name)
tokenizer.pad_token = tokenizer.eos_token

print("Lade das Basis-Modell herunter (das kann beim ersten Mal kurz dauern)...")
modell = AutoModelForCausalLM.from_pretrained(
    modell_name,
    quantization_config=schrumpf_einstellungen,
    device_map="auto" # Das sucht automatisch deine RTX 4060 Ti!
)

# 4. Das "Effektgerät" (LoRA) einstellen
# Wir frieren das große Gehirn ein und trainieren nur dieses kleine Zusatzteil.
lora_einstellungen = LoraConfig(
    r=16, # Vorher 8. Ein höherer Rang erlaubt der KI, komplexere Zusammenhänge zu speichern.
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Wir trainieren jetzt mehr "Schaltzentren" im Hirn
    task_type="CAUSAL_LM"
)

# 5. Wie sollen die Daten aussehen?
def text_formatieren(beispiel):
    return f"Frage: {beispiel['frage']}\nAntwort: {beispiel['antwort']}"

# 6. Die Trainingsregeln (Der Motor)
trainings_regeln = TrainingArguments(
    output_dir="./mein_qwen_pirat",
    per_device_train_batch_size=1, 
    learning_rate=3e-5,            # Etwas niedriger für mehr Stabilität
    num_train_epochs=28,           
    logging_steps=1,
    # --- DIESE DREI ZEILEN SIND JETZT ENTSCHEIDEND ---
    fp16=False,                    # Wir schalten es aus
    bf16=False,                    # Auch aus
    gradient_checkpointing=True,   # Spart Speicherplatz auf andere Weise
    # ------------------------------------------------
    save_strategy="epoch"
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

# 8. Das fertige Modell (das angelernte Effektgerät) speichern
trainer.model.save_pretrained("./mein_fertiger_pirat")
print("\nArrr! Training beendet! Dein Modell wurde im Ordner 'mein_fertiger_pirat' gespeichert.")