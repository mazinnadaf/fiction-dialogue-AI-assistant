import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Chat function
def chat_fn(message, history):
    # history is already a list of {"role": ..., "content": ...}
    messages = history + [{"role": "user", "content": message}]

    # Apply chat template (for instruct models)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt echo
    if reply.startswith(input_text):
        reply = reply[len(input_text):].strip()

    return reply

# Custom CSS for purple gradient theme + chat bubbles
custom_css = """
body {
    background: linear-gradient(135deg, #2b0033, #3d0066, #5a0099);
    color: white;
}
.gradio-container {
    background: transparent !important;
}
.gr-chatbot {
    background-color: rgba(0, 0, 0, 0.3) !important;
    border-radius: 12px;
    padding: 10px;
}
.gr-chatbot .message.user {
    background: linear-gradient(135deg, #7b2cbf, #9d4edd);
    color: white !important;
    border-radius: 12px;
    padding: 8px 12px;
}
.gr-chatbot .message.bot {
    background: linear-gradient(135deg, #4a148c, #6a1b9a);
    color: white !important;
    border-radius: 12px;
    padding: 8px 12px;
}
.gr-textbox textarea {
    background-color: rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    border-radius: 8px;
}
"""

# ChatInterface
chatbot = gr.ChatInterface(
    fn=chat_fn,
    chatbot=gr.Chatbot(type="messages"),  # ✅ use OpenAI-style messages
    textbox=gr.Textbox(placeholder="Type your message here..."),
    title="Fiction Dialogue Writing Assistant",
    description="Finetuning SmolLM-360M-Instruct and custom language model for creative writing dialogue generation - work in progress - code and citations at: [GitHub](https://github.com/mazinnadaf/fiction-dialogue-AI-assistant)",
    css=custom_css
)

if __name__ == "__main__":
    chatbot.launch()
