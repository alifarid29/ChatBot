"""
Bookshop Chatbot – Standalone Tkinter GUI
Run:  python chatbot_ui.py
Requires the trained model (bookshop_chatbot_model.h5), words.pkl,
classes.pkl and the intents (1).json / data.csv to be in the same folder.
"""

import json
import pickle
import random
import textwrap
import tkinter as tk
from tkinter import scrolledtext, ttk

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

# ── Load artifacts ─────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()

MODEL_PATH   = "bookshop_chatbot_model.h5"
WORDS_PATH   = "words.pkl"
CLASSES_PATH = "classes.pkl"
INTENTS_PATH = "intents (1).json"
DATA_PATH    = "data.csv"

model    = load_model(MODEL_PATH)
words    = pickle.load(open(WORDS_PATH,   "rb"))
classes  = pickle.load(open(CLASSES_PATH, "rb"))
books_df = pd.read_csv(DATA_PATH)

with open(INTENTS_PATH, encoding="utf-8") as f:
    intents = json.load(f)

CATEGORY_TAGS = {
    i["tag"] for i in intents["intents"]
    if i["tag"] not in ("greeting", "goodbye", "thanks", "book_search")
}

CONFIDENCE_THRESHOLD = 0.35

# ── NLP helpers ────────────────────────────────────────────────────────────────

def clean_up_sentence(sentence):
    tokens = word_tokenize(sentence)
    return [lemmatizer.lemmatize(t.lower()) for t in tokens]


def bow(sentence):
    token_list = clean_up_sentence(sentence)
    return np.array(
        [1 if w in token_list else 0 for w in words], dtype=np.float32
    )


def predict_class(sentence):
    vec = bow(sentence)
    res = model.predict(vec.reshape(1, -1), verbose=0)[0]
    results = [
        {"intent": classes[i], "probability": float(res[i])}
        for i in range(len(res))
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def get_csv_book(category_tag, top_n=5):
    mask   = books_df["categories"].str.contains(category_tag, case=False, na=False)
    subset = books_df[mask].dropna(subset=["average_rating"])
    if subset.empty:
        return None
    book   = subset.nlargest(top_n, "average_rating").sample(1).iloc[0]
    title  = book["title"]
    author = book.get("authors", "Unknown")
    rating = book.get("average_rating", "N/A")
    desc   = str(book.get("description", "")).strip()
    desc   = textwrap.shorten(desc, width=350, placeholder="...")
    return f"Book: {title}\nAuthor: {author}\nRating: {rating}/5\n\n{desc}"


def get_intent_response(intents_list):
    if not intents_list:
        return "I'm not sure I understand. Could you rephrase that?"
    tag = intents_list[0]["intent"]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            chosen = random.choice(intent["responses"])
            if isinstance(chosen, dict):
                return (
                    f"Book: {chosen.get('Book', '?')}\n"
                    f"Rating: {chosen.get('Rate', 'N/A')}/5\n\n"
                    f"{chosen.get('Feedback', '')}"
                )
            return chosen
    return "I'm not sure I understand. Could you rephrase that?"


def chatbot_response(user_message):
    intents_list = predict_class(user_message)
    if not intents_list:
        return "I'm not confident enough to answer that. Could you rephrase?"

    top = intents_list[0]
    conf = top["probability"]
    tag  = top["intent"]

    if conf < CONFIDENCE_THRESHOLD:
        return (
            f"I'm not confident enough to answer that "
            f"(confidence: {conf:.1%} < {CONFIDENCE_THRESHOLD:.0%}).\n"
            "Could you rephrase or ask about a specific book genre?"
        )

    if tag in CATEGORY_TAGS:
        csv_resp = get_csv_book(tag)
        if csv_resp:
            return f"[Live Recommendation – {tag}]\n\n{csv_resp}"

    return get_intent_response(intents_list)


# ── Tkinter UI ─────────────────────────────────────────────────────────────────

BG_DARK   = "#1e1e2e"
BG_PANEL  = "#2a2a3e"
BG_INPUT  = "#313145"
FG_WHITE  = "#cdd6f4"
FG_BLUE   = "#89b4fa"
FG_GREEN  = "#a6e3a1"
FG_PURPLE = "#cba6f7"
ACCENT    = "#89dceb"

class BookshopChatbotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bookshop Chatbot")
        self.geometry("700x600")
        self.minsize(500, 450)
        self.configure(bg=BG_DARK)
        self._build_ui()
        self._greet()

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self, bg=FG_PURPLE, height=56)
        header.pack(fill="x")
        tk.Label(
            header,
            text="  Bookshop Chatbot",
            bg=FG_PURPLE,
            fg=BG_DARK,
            font=("Helvetica", 16, "bold"),
            anchor="w",
        ).pack(side="left", padx=12, pady=12)
        tk.Label(
            header,
            text="Powered by NLP + TensorFlow",
            bg=FG_PURPLE,
            fg=BG_DARK,
            font=("Helvetica", 9),
        ).pack(side="right", padx=12)

        # ── Chat area ──
        chat_frame = tk.Frame(self, bg=BG_DARK)
        chat_frame.pack(fill="both", expand=True, padx=12, pady=(10, 4))

        self.chat_area = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            state="disabled",
            bg=BG_PANEL,
            fg=FG_WHITE,
            font=("Consolas", 11),
            relief="flat",
            bd=0,
            padx=10,
            pady=8,
            cursor="arrow",
        )
        self.chat_area.pack(fill="both", expand=True)

        # Configure text tags for colored bubbles
        self.chat_area.tag_config("user_tag",  foreground=FG_BLUE,   font=("Consolas", 10, "bold"))
        self.chat_area.tag_config("user_msg",  foreground=FG_WHITE)
        self.chat_area.tag_config("bot_tag",   foreground=FG_GREEN,  font=("Consolas", 10, "bold"))
        self.chat_area.tag_config("bot_msg",   foreground=FG_WHITE)
        self.chat_area.tag_config("divider",   foreground="#45475a")

        # ── Quick-reply buttons ──
        quick_frame = tk.Frame(self, bg=BG_DARK)
        quick_frame.pack(fill="x", padx=12, pady=(0, 4))

        quick_replies = [
            "Hello!", "Recommend Fiction", "Science fiction",
            "History book", "Goodbye",
        ]
        for qr in quick_replies:
            btn = tk.Button(
                quick_frame,
                text=qr,
                bg=BG_INPUT,
                fg=ACCENT,
                font=("Helvetica", 9),
                relief="flat",
                padx=8,
                pady=4,
                cursor="hand2",
                activebackground=FG_PURPLE,
                activeforeground=BG_DARK,
                command=lambda msg=qr: self._quick_send(msg),
            )
            btn.pack(side="left", padx=3)

        # ── Input row ──
        input_frame = tk.Frame(self, bg=BG_DARK)
        input_frame.pack(fill="x", padx=12, pady=(0, 12))

        self.entry = tk.Entry(
            input_frame,
            bg=BG_INPUT,
            fg=FG_WHITE,
            insertbackground=FG_WHITE,
            font=("Helvetica", 12),
            relief="flat",
            bd=0,
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=10, padx=(0, 8))
        self.entry.bind("<Return>", lambda e: self._send())
        self.entry.focus_set()

        send_btn = tk.Button(
            input_frame,
            text="Send",
            bg=FG_PURPLE,
            fg=BG_DARK,
            font=("Helvetica", 11, "bold"),
            relief="flat",
            padx=18,
            pady=8,
            cursor="hand2",
            activebackground=FG_BLUE,
            command=self._send,
        )
        send_btn.pack(side="right")

    def _append(self, tag_name, tag_text, msg_tag, message):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", f"{tag_text}\n", tag_name)
        self.chat_area.insert("end", f"{message}\n\n", msg_tag)
        self.chat_area.config(state="disabled")
        self.chat_area.see("end")

    def _greet(self):
        greeting = chatbot_response("Hello")
        self._append("bot_tag", "Bookshop Bot", "bot_msg", greeting)

    def _send(self):
        user_msg = self.entry.get().strip()
        if not user_msg:
            return
        self.entry.delete(0, "end")
        self._append("user_tag", "You", "user_msg", user_msg)
        response = chatbot_response(user_msg)
        self._append("bot_tag", "Bookshop Bot", "bot_msg", response)

    def _quick_send(self, msg):
        self.entry.delete(0, "end")
        self.entry.insert(0, msg)
        self._send()


if __name__ == "__main__":
    app = BookshopChatbotApp()
    app.mainloop()
