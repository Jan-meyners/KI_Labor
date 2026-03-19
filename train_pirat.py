import torch
import json
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# Hier geht es dann mit dem Rest deines Codes weiter...
print("Lade Werkzeuge für das große 8B KI-Training... Bitte warten!")

# --- WICHTIG: Trag hier den exakten Namen deiner neuen generierten Datei ein! ---
daten_pfad = "jack_massive_8b_dataset.json" 

if not os.path.exists(daten_pfad):
    print(f"FEHLER: Die Datei '{daten_pfad}' wurde nicht gefunden!")
    exit()

# 1. Das "Schulbuch" laden
with open(daten_pfad, "r", encoding="utf-8") as f:
    daten = json.load(f)
mein_schulbuch = Dataset.from_list(daten)
print(f"✅ Erfolgreich {len(daten)} Fragen (inkl. Variationen) geladen.")

# 2. Der "Schrumpfstrahler" (4-Bit Quantisierung für 8GB VRAM)
# Lebenswichtig für Llama 8B auf einer RTX 4060 Ti!
schrumpf_einstellungen = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # Turbo-Boost für RTX 4000er
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 3. Das neue Basis-Modell (Llama 3.1 - 8B Instruct) laden
modell_name = "unsloth/Meta-Llama-3.1-8B-Instruct" 

tokenizer = AutoTokenizer.from_pretrained(modell_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Lade das riesige {modell_name} herunter... (Das ist 8B, dauert also kurz!)")
modell = AutoModelForCausalLM.from_pretrained(
    modell_name,
    quantization_config=schrumpf_einstellungen,
    device_map="auto" 
)

# 4. Das "Effektgerät" (LoRA) einstellen - Maximales Verständnis!
lora_einstellungen = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    task_type="CAUSAL_LM",
    bias="none"
)

# 5. DIE NEUE FORMATIERUNG (Fehler behoben!)
# Jetzt bauen wir System-Prompt (instruction), Frage (input) und Antwort (output) sauber zusammen:
def text_formatieren(beispiel):
    return f"{beispiel['instruction']}\n\n### Frage:\n{beispiel['input']}\n\n### Antwort:\n{beispiel['output']}"

# 6. Die Trainingsregeln (Wieder kugelsicher und OHNE den nervigen Längen-Begrenzer)
trainings_regeln = TrainingArguments(
    output_dir="./professor_jack_8b_checkpoints", 
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8, 
    learning_rate=2e-5,            
    num_train_epochs=3,            
    logging_steps=10,              
    bf16=True,                     
    fp16=False,                    
    gradient_checkpointing=True,   
    save_strategy="epoch",         
    optim="paged_adamw_8bit"
    # max_seq_length haben wir hier komplett gelöscht!
)

# Alles zusammenbauen
trainer = SFTTrainer(
    model=modell,
    train_dataset=mein_schulbuch,
    peft_config=lora_einstellungen,
    formatting_func=text_formatieren,
    args=trainings_regeln
    # Auch hier ist max_seq_length komplett weg!
)

# 7. Startschuss!
print("\n🚀 --- Mache die Leinen los... Das ultimative 8B-Training beginnt! --- 🚀")
print("Hinweis: Da das Modell so groß ist und wir fast 4.000 Fragen haben, wird dieses Training einige Stunden dauern.")
trainer.train()

# 8. Das absolut finale Modell speichern
endgueltiger_pfad = "./professor_jack_8b_fertig"
trainer.model.save_pretrained(endgueltiger_pfad)
tokenizer.save_pretrained(endgueltiger_pfad)
print(f"\n🎉 TRAINING BEENDET! 🎉")
print(f"Dein massiver 'Professor Jack (8B)' wurde im Ordner '{endgueltiger_pfad}' gespeichert.")