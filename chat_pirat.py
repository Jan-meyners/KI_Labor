import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("Hole den Medien-Experten ins Büro... (Lade Modelle)")

modell_name = "Qwen/Qwen2.5-1.5B"

# 1. Die Lego-Stein-Maschine laden
tokenizer = AutoTokenizer.from_pretrained(modell_name)

# 2. Den Schrumpfstrahler wieder einschalten
schrumpf_einstellungen = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 3. Das "nackte" Basis-Modell laden
basis_modell = AutoModelForCausalLM.from_pretrained(
    modell_name,
    quantization_config=schrumpf_einstellungen,
    device_map="auto"
)

# 4. DAS MAGISCHE Puzzleteil: Dein angelerntes Effektgerät anstöpseln!
# HINWEIS: Wenn du den Ordnernamen beim Training geändert hast, musst du ihn hier anpassen!
experten_modell = PeftModel.from_pretrained(basis_modell, "./mein_fertiger_pirat")

print("\n--- Der Medien-Experte ist wach und bereit! ---")

# 5. Unser Chat-Programm
while True:
    deine_frage = input("\nDeine Frage an den Experten (oder 'ende' zum Abbrechen): ")
    
    if deine_frage.lower() == 'ende':
        break
        
    prompt = f"Frage: {deine_frage}\nAntwort:"
    
    eingabe = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # HIER IST DAS UPDATE!
    ausgabe_tokens = experten_modell.generate(
        **eingabe, 
        max_new_tokens=150,     # Wir erlauben längere Experten-Antworten (vorher 40)
        do_sample=True,         # Der Hauptschalter für Kreativität! Warnung verschwindet.
        temperature=0.3,        # 0.3 ist gut für Experten (weniger Halluzinationen, mehr Fakten)
        pad_token_id=tokenizer.eos_token_id
    )
    
    fertiger_text = tokenizer.decode(ausgabe_tokens[0], skip_special_tokens=True)
    reine_antwort = fertiger_text.replace(prompt, "").strip()
    
    print(f"\n🎓 Experte sagt: {reine_antwort}")