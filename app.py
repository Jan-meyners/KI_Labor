import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# 1. Seite konfigurieren
st.set_page_config(page_title="Professor Jack - Medienpädagoge", page_icon="🎓")
st.title("🎓 Professor Jack")
st.markdown("Dein KI-Experte für Medienpädagogik. Frag mich einfach alles zu Studien oder Erziehung!")

# 2. Modell & Adapter laden (mit Cache, damit es nur einmal lädt)
@st.cache_resource
def load_everything():
    base_model_name = "unsloth/Llama-3.2-3B-Instruct"
    adapter_path = "./professor_jack_fertig"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # 4-Bit Konfiguration für deine RTX 4060 Ti
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Basis-Modell laden
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    # Deine trainierten Adapter laden
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return tokenizer, model

# 3. App-Logik starten
try:
    with st.spinner("Professor Jack schlägt seine Bücher auf..."):
        tokenizer, model = load_everything()
    
    # Chat-Historie im Session State speichern
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat-Verlauf auf der Seite anzeigen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Eingabe-Feld für den User
    if prompt := st.chat_input("Schreib mir eine Nachricht..."):
        # User-Nachricht speichern und anzeigen
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Antwort generieren
        with st.chat_message("assistant"):
            # System-Anweisung: Wer ist Jack und wie soll er sich verhalten?
            system_instruction = (
                "Du bist Professor Jack, ein empathischer Medienpädagoge. "
                "Antworte fachlich fundiert auf Basis der JIM-Studie, bleibe freundlich und beratend. "
                "Du musst dich nicht in jeder Nachricht förmlich begrüßen, wenn es ein laufendes Gespräch ist."
            )
            
            # Kontext aufbauen: Die letzten 3 Nachrichten mitgeben
            context = ""
            if len(st.session_state.messages) > 1:
                for msg in st.session_state.messages[-4:-1]: # Letzte 3 Nachrichten
                    role = "User" if msg["role"] == "user" else "Jack"
                    context += f"{role}: {msg['content']}\n"

            # Finaler Prompt im Trainings-Format
            full_prompt = (
                f"### System:\n{system_instruction}\n\n"
                f"{context}"
                f"### Frage:\n{prompt}\n\n"
                f"### Antwort:\n"
            )
            
            inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=400, 
                    temperature=0.6,    # Etwas mehr Abwechslung in der Sprache
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Antwort aus dem Text extrahieren
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Wir splitten beim letzten "### Antwort:", um nur die neue Antwort zu bekommen
            clean_response = full_text.split("### Antwort:")[-1].strip()
            

            # Antwort anzeigen und speichern
            st.markdown(clean_response)
            st.session_state.messages.append({"role": "assistant", "content": clean_response})

    # --- HIER WAR DER FEHLER: Die Sidebar muss NOCH IM try-Block stehen ---
    with st.sidebar:
        st.subheader("Optionen")
        if st.button("Chat-Verlauf löschen"):
            st.session_state.messages = []
            st.rerun()

except Exception as e:
    st.error(f"Kritischer Fehler: {e}")