import tkinter as tk
from langid.langid import LanguageIdentifier, model

def get_cefr_level(text):
    # Replace this function with your actual CEFR level determination logic
    # For now, let's assume a simple logic based on text length
    if len(text) < 50:
        return "A1"
    elif len(text) < 100:
        return "A2"
    elif len(text) < 200:
        return "B1"
    elif len(text) < 300:
        return "B2"
    else:
        return "C1"

def on_submit():
    user_text = text_entry.get("1.0", "end-1c")
    lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    lang, _ = lang_identifier.classify(user_text)

    if lang == 'fr':  # Change 'en' to your desired language code
        cefr_level = get_cefr_level(user_text)
        result_label.config(text=f"CEFR Level: {cefr_level}")
    else:
        result_label.config(text="CEFR Level determination only supported for English text.")

# Create the main window
root = tk.Tk()
root.title("CEFR Level Detector")

# Create and place the text entry
text_entry = tk.Text(root, height=10, width=50)
text_entry.pack(pady=10)

# Create and place the submit button
submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack(pady=10)

# Create and place the result label
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Start the main loop
root.mainloop()
