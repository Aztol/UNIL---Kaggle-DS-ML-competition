import streamlit as st
import torch
from transformers import CamembertTokenizer, AutoModelForSequenceClassification
import numpy as np
from langdetect import detect

# Setup code for difficulty mapping, model and tokenizer paths
difficulty_mapping = {
    0: 'A1',
    1: 'A2',
    2: 'B1',
    3: 'B2',
    4: 'C1',
    5: 'C2'
}

model_path = '/Users/Gaetan_1/Documents/Repository/UNIL---Kaggle-DS-ML-competition/model_fromage'  # Replace with the actual path to your model
tokenizer_path = '/Users/Gaetan_1/Documents/Repository/UNIL---Kaggle-DS-ML-competition/model_fromage/tokenizer'  # Replace with the actual path to your tokenizer

tokenizer = CamembertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

MAX_LEN = 512

st.title("CEFR Level Detector - UNIL_IWC")
intro_text = """
Welcome to **CEFR Level Detector**! Simply input your text or upload a document, and we'll quickly assess your French language level based on European standards. Our platform offers a straightforward, no-fuss approach to understanding your proficiency in French. Get your language evaluation in an instant!
"""
st.markdown(intro_text, unsafe_allow_html=False)
user_text = st.text_area("Enter your text here :", height=250)

st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Or choose a text file (.txt) to analyze :", type="txt")

st.markdown("<br>", unsafe_allow_html=True)

# Display an initial progress bar with a cursor at the starting position
cursor_position = '0%'
st.markdown("""
    <!-- Custom HTML and CSS for the progress bar goes here -->
    <style>
        .progress-bar-container {
            width: 100%;
            background-color: #e6e6e6;
            border-radius: 13px;
            height: 26px;
            position: relative;
            display: flex;
            align-items: center;
        }
        .progress-bar-fill {
            width: 100%;
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(to right, #ff2d00, #ff7700, #ffbc00, #ade300, #4cd000);
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }
        .progress-bar-label {
            position: absolute;
            z-index: 2;
            width: 100%;
            display: flex;
            justify-content: space-between;
            padding: 0 10px;
            font-size: 14px;
        }
        .progress-bar-label div {
            text-align: center;
            color: black;
            font-weight: bold;
            user-select: none;
        }
           .progress-bar-cursor {
                width: 0;
                height: 0;
                border-left: 8px solid transparent;
                border-right: 8px solid transparent;
                border-top: 16px solid black;
                position: absolute;
                top: 0;
                left: 50%;
                transform: translate(-50%, -100%);
                z-index: 3;
            }
            .progress-bar-cursor::after {
                content: "";
                position: absolute;
                top: 2px;
                left: -8px;
                width: 0;
                height: 0;
                border-left: 8px solid transparent;
                border-right: 8px solid transparent;
                border-top: 16px solid transparent;
            }
    </style>
    <div class="progress-bar-container">
        <div class="progress-bar-fill"></div>
        <div class="progress-bar-cursor" id="cursor"></div>
        <div class="progress-bar-label">
            <div>A1</div><div>A2</div><div>B1</div><div>B2</div><div>C1</div><div>C2</div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Submit"):
    if uploaded_file is not None:
        text_to_analyze = str(uploaded_file.read(), "utf-8")
    elif user_text:
        text_to_analyze = user_text
    else:
        text_to_analyze = ''
        st.error("Please enter a text or upload a file before submitting.")

    if text_to_analyze:
        try:
            if detect(text_to_analyze) != 'fr':
                st.error("Please submit a text in French")
            else:
                input_ids = tokenizer.encode(text_to_analyze, add_special_tokens=True, max_length=MAX_LEN, padding='max_length', truncation=True)
                attention_masks = [float(i > 0) for i in input_ids]

                input_ids = torch.tensor([input_ids])
                attention_masks = torch.tensor([attention_masks])

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_masks)

                logits = outputs.logits
                logits = logits.detach().cpu().numpy()
                predicted_label_index = np.argmax(logits, axis=1).flatten()[0]
                predicted_label = difficulty_mapping[predicted_label_index]

                #st.success(f"Prédiction : {predicted_label}")
                
                # Calculate the position of the cursor based on the predicted level
                cursor_position = {
                    'A1': '2%',
                    'A2': '22%',
                    'B1': '40%',
                    'B2': '60%',
                    'C1': '78%',
                    'C2': '98%'
                }[predicted_label]

                # Update the progress bar with the new cursor position
                st.markdown(f"<style>.progress-bar-cursor {{ left: {cursor_position}; }}</style>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}")