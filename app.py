import spaces
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

with open("vocab.pkl","rb") as f:
    vocab=pickle.load(f)
with open("label2id.pkl","rb") as f:
    label2id=pickle.load(f)
with open("config.pkl","rb") as f:
    config=pickle.load(f)

id2label={v:k for k,v in label2id.items()}
EMBEDDING_DIM=config["embedding_dim"]
HIDDEN_DIM=config["hidden_dim"]
MAX_LENGTH=config["max_length"]

def tokenize(text):
    text=str(text).lower()
    text=re.sub(r"[^a-z0-9 ]+"," ",text)
    return text.split()

def encode(text):
    tokens = tokenize(text)

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]
    ids = ids[:MAX_LENGTH]
    return ids

def pad_sequence(sequence):
    if len(sequence) >= MAX_LENGTH:
        return sequence[:MAX_LENGTH]
    return sequence + [vocab["<PAD>"]] * (MAX_LENGTH - len(sequence))

def build_mcq_text(prompt,option):
    return f"{prompt} [SEP] {option}"

class BiLSTMMultipleChoice(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()    #Initialization
        #Word Embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=vocab["<PAD>"]
        )
        #BiLSTM Encoder
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM, 
            hidden_size=HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        #Dropout
        self.dropout = nn.Dropout(0.3)
        #Classifier
        self.classifier = nn.Linear(HIDDEN_DIM * 2, 1)

    def forward(self, input_ids):
        batch_size, num_choices, seq_len = input_ids.shape

        input_ids = input_ids.view(batch_size * num_choices, seq_len)
        #Embeddings
        embeddings = self.embedding(input_ids)
        outputs, _ = self.lstm(embeddings)
        #Masked Mean Pooling (Ignores the Padding and averages only valid tokens)
        mask = (input_ids != vocab["<PAD>"]).unsqueeze(-1)  #Ignoring <PAD>
        mask = mask.float()            #Converting from boolean to float
        outputs = outputs * mask       #Zeroes out padded positions
        summed = outputs.sum(dim=1)    #Summing up only vvalid tokens
        lengths = mask.sum(dim=1)      #Counting valid tokens
        lengths = lengths.clamp(min=1)
        pooled = summed / lengths
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        logits = logits.view(batch_size,num_choices)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=BiLSTMMultipleChoice(len(vocab))
state_dict = torch.load(
    "best_bilstm_model.pt",
    map_location="cpu"
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

@spaces.GPU
def predict(prompt, a, b, c, d, e):
    choices = [
        pad_sequence(
            encode(build_mcq_text(prompt, option))
        )
        for option in [a, b, c, d, e]
    ]
    input_ids = torch.tensor([choices], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_ids)
        probs = F.softmax(logits, dim=1)[0]

    values, indices = torch.topk(probs, 3)

    prediction = id2label[indices[0].item()]

    top3 = "\n".join(
        [
            f"{id2label[idx.item()]} : {value.item():.4f}"
            for value, idx in zip(values, indices)
        ]
    )
    return prediction, top3

demo=gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Option A"),
        gr.Textbox(label="Option B"),
        gr.Textbox(label="Option C"),
        gr.Textbox(label="Option D"),
        gr.Textbox(label="Option E"),
    ],
    outputs=[gr.Textbox(label="Prediction"),gr.Textbox(label="Top-3")],
    title="Smart MCQ Solver - BiLSTM"
)

if __name__=="__main__":
    demo.launch()
