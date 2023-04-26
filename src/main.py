from keras.models import load_model
import numpy as np
import pandas as pd
from typing import Annotated
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')


# Load the saved model from the HDF5 file
model = load_model("./models/ner_model.h5")

# Load the dataset as a pandas DataFrame
df = pd.read_csv("./data/ner_dataset.csv", encoding="latin1")

# Filter out unnecessary columns
df = df.drop(columns=["POS"])

# Rename columns to match CoNLL-2003 format
df = df.rename(columns={"Sentence #": "Sentence", "Tag": "NE"})

# Replace NaN values with the string "O"
df = df.fillna("O")

# Define the input and output dimensions
n_words = len(df["Word"].unique())
n_tags = len(df["NE"].unique())


def preprocess_sentence(sentence, word_to_int, max_len):
    sentence = [word_to_int.get(word, 0) for word in sentence.split()]
    sentence = sentence + [0] * (max_len - len(sentence))
    return np.array(sentence)


def predict_entities(sentences):
    # Convert the words to numerical values
    word2idx = {w: i + 1 for i, w in enumerate(df["Word"].unique())}
    word2idx["PAD"] = 0
    word2idx["UNK"] = n_words + 1
    tag2idx = {t: i for i, t in enumerate(df["NE"].unique())}
    idx2word = {i: w for w, i in word2idx.items()}
    idx2tag = {i: t for t, i in tag2idx.items()}

    # Preprocess the new sentences
    X_test = np.array([preprocess_sentence(sentence, word2idx, 10)
                      for sentence in sentences])

    # Make predictions on the new sentences
    y_pred = model.predict(X_test)

    # Convert the predicted tags to named entities
    int_to_tag = {i: t for t, i in tag2idx.items()}
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = [[int_to_tag[i] for i in sentence] for sentence in y_pred]

    # Print the predicted named entities for each sentence
    return {sentences[0]: y_pred[0][:len(sentences[0].split())]}
#   for i, sentence in enumerate(sentences):
#       print(f"Sentence {i+1}:")
#       print(sentence)
#       print("Predicted named entities:")
#       print(y_pred[i][:len(sentence.split())])
#       print()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse('home.html', {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: Annotated[str, Form()]):

    tags = [x[1] for x in enumerate(df["NE"].unique())]
    colors1 = ["#dc2626", "#ea580c", "#65a30d", "#16a34a", "#059669", "#0d9488",
               "#0891b2", "#2563eb", "#4f46e5", "#7c3aed", "#3b0764", "#c026d3", "#db2777", "#0f172a", "#831843", "#4a044e", "#022c22"]
    colors = {tags[i]: color for i, color in enumerate(colors1)}

    pred = predict_entities([text])
    pred = enumerate(list(pred.values())[0])
    tokens = text.split()
    # for t in enumerate(df["NE"].unique()):
    #     text += "\n\n" + t[1] {% comment %} <p style="padding: 10px; border-radius: 10px; color: {{colors[i]}}; background: {{colors[i]+"55"}}">{{tag}}</p> {% endcomment %}

    return templates.TemplateResponse("predict.html", {"request": request, "colors": colors, "tokens": tokens, "pred": pred})
