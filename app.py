import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Seite einrichten
st.set_page_config(page_title="Medien-Experte KI", page_icon="🛡️")
st.title("🛡️ Dein Medien-Experte")
st.markdown("Frag mich alles zu KIM/JIM-Studien, Gesetzen und Erziehungstipps.")

# Modell und Adapter laden
@st.cache_resource
def load_everything():
    # 1. Das Basis-Modell (muss dasselbe wie beim Training sein)
    # ÄNDERE DIESE ZEILE in deiner app.py:
    # Versuche es mit diesem exakten Namen (der hat die Größe 1536):
    base_model_name = "Qwen/Qwen2.5-1.5B"
    
    # 2. Dein Adapter-Ordner (der kleine Ordner)
    adapter_path = "./mein_fertiger_pirat"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Basis-Modell laden
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Deine gelernten Adapter (die paar MB) darüberladen
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return tokenizer, model

try:
    with st.spinner("Experten-Wissen wird geladen... bitte kurz warten."):
        tokenizer, model = load_everything()
    
    # Chat-Verlauf initialisieren
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Verlauf anzeigen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Eingabe-Feld
# Eingabe-Feld
    if prompt := st.chat_input("Stelle hier deine Frage..."):
        # 1. Deine Frage im Speicher ablegen
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. Deine Frage sofort anzeigen
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. Antwort generieren
        with st.chat_message("assistant"):
            messages = [
                {"role": "system", "content": "Du bist ein hilfreicher Medien-Experte. Antworte präzise auf Basis deines Wissens."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                # Wir fügen terminators hinzu, damit das cliông-Gerede aufhört
                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    tokenizer.convert_tokens_to_ids("<|endoftext|>")
                ]
                
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=True, 
                    temperature=0.3,    # Bleib bei der Wahrheit
                    top_p=0.9,
                    repetition_penalty=1.2, # Erhöht auf 1.2 gegen das cliông-Gerede
                    eos_token_id=terminators, # Sag der KI, wann Schluss ist
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Die Antwort sauber herausschneiden
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
            clean_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Antwort anzeigen und speichern
            st.markdown(clean_response)
            st.session_state.messages.append({"role": "assistant", "content": clean_response})

except Exception as e:
    st.error(f"Fehler beim Starten: {e}")
    st.info("Stelle sicher, dass peft installiert ist: pip install peft")