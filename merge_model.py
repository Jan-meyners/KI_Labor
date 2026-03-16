import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Pfade festlegen
# Versuche es mit dieser Version (das ist die kleinere 1.1B Version, die zu 1536 passt)
base_model_path = "Qwen/Qwen1.5-1.1B-Chat"
adapter_path = "./mein_fertiger_pirat"
save_path = "./mein_experte_VOLLVERSION" # Hier wird das große Modell gespeichert

print("Lade Basis-Modell...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu" # Wir nutzen CPU zum Mergen, um VRAM-Fehler zu vermeiden
)

print("Lade Adapter und verbinde sie...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# Jetzt werden die Gewichte fest miteinander verschmolzen
print("Verschmelze Gewichte (Merging)...")
model = model.merge_and_unload()

print(f"Speichere Vollversion in {save_path}...")
model.save_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(save_path)

print("Fertig! Du hast jetzt ein eigenständiges, großes Modell.")