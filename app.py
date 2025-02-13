import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load FLAN-T5 model & tokenizer
MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Generate a response using FLAN-T5
def generate_response(user_input):
    """Generate a detailed medical response."""
    input_text = f"Medical expert response: {user_input}"

    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=350,  # Increased response length as our need
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response if response else "I'm sorry, I couldn't process that. Please try again."


# Streamlit UI
st.set_page_config(page_title="Healthcare Chatbot", page_icon="🤖", layout="centered")

# Custom styling
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stTextArea>div>div>textarea {
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }
        .stMarkdown {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("💬 Healthcare Assistant Chatbot 🏥")

# Initialize 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input area
user_input = st.text_area("📝 Enter your medical question:", height=100)

# Submit button
if st.button("💡 Get Advice"):
    if user_input.strip():
        with st.spinner("🤖 Generating response..."):
            response = generate_response(user_input)
        
        # Store conversation in session state
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Healthcare Assistant", response))
    
    else:
        st.warning("Please enter a question.")

# Display chat history
st.markdown("## 🗨️ Chat History")
for role, text in st.session_state.chat_history:
    with st.chat_message("assistant" if role == "Healthcare Assistant" else "user"):
        st.markdown(f"**{role}:** {text}")

