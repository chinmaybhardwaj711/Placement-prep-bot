---
title: Placement Prep Bot
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.45.0
app_file: app.py
pinned: false
---

# 🎯 Placement Prep Bot

A RAG-based AI assistant that answers placement interview questions from your own study material — not from the internet, not from generic training data, but from the notes you actually studied.

Upload your PDFs (or even handwritten notes), pick a subject, and ask away.

---

## Why I built this

Most AI chatbots answer placement questions from their general training. That's fine, but it means the answers aren't grounded in *your* syllabus, *your* professor's notes, or the specific way your college teaches a topic.

This bot fixes that. You bring the material, it becomes the source of truth.

---

## How it works

The pipeline is straightforward:

1. You upload one or more PDFs
2. The app extracts text (or runs OCR if the PDF is handwritten/scanned)
3. Text is split into chunks and embedded using `all-MiniLM-L6-v2`
4. Embeddings are stored in a FAISS vector index
5. When you ask a question, the top relevant chunks are retrieved using MMR search
6. Those chunks are passed to Llama 3.1 (via Groq) along with a subject-specific prompt
7. The answer comes back grounded in your material — with source citations

The bot also remembers your previous questions within a session, so you can ask follow-ups naturally.

---

## Features

- **Multi-PDF support** — upload multiple documents at once, query across all of them
- **Handwritten notes via OCR** — uses EasyOCR to extract text from scanned or photographed notes
- **Conversational memory** — follow-up questions work, the bot remembers context
- **Source citations** — every answer tells you which PDF it came from
- **Retrieval transparency** — expand any answer to see the exact chunks that were retrieved, with similarity scores
- **Subject modes** — switch between DSA, OOPS, DBMS, OS, CN, and General to get domain-appropriate explanations
- **PDF hash caching** — same PDFs won't be re-embedded on re-upload, saves time

---

## Tech stack

| Component | Tool |
|---|---|
| LLM | Llama 3.1 8B via Groq API |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector store | FAISS |
| OCR | EasyOCR |
| Orchestration | LangChain |
| UI | Streamlit |
| Deployment | Hugging Face Spaces |

---

## Running locally

```bash
git clone https://github.com/Chinmay711/placement-prep-bot
cd placement-prep-bot

python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

Then run:
```bash
streamlit run app.py
```

---

## Using the app

1. Upload your study PDFs in the sidebar (typed or handwritten)
2. Click **⚡ Process PDFs** and wait for indexing to finish
3. Select a subject mode (DSA, OOPS, DBMS, OS, CN, or General)
4. Ask any placement interview question in the chat
5. Expand **🔍 View Retrieved Chunks** to see exactly what the bot used to answer

---

## Evaluation

Ran a keyword-based evaluation across DSA, OOPS, and DBMS subject PDFs:

| Question | Subject | Score |
|---|---|---|
| What is the time complexity of binary search? | DSA | 67% |
| What is polymorphism? | OOPS | 67% |
| What are ACID properties? | DBMS | 100% |

Scoring checks whether expected keywords appear in the answer. Full results in `eval_results.json`.

---

## Engineering decisions worth knowing

**Why MMR over simple similarity search?**
MMR (Maximum Marginal Relevance) balances relevance with diversity. Without it, the top 5 retrieved chunks often end up being nearly identical — just slightly different parts of the same paragraph. MMR forces diversity across the retrieved set, giving the LLM more context to work with.

**Why FAISS over ChromaDB?**
FAISS is faster for pure similarity search and has no server dependency. For a stateless deployment like HF Spaces where everything is rebuilt per session, FAISS is the right call.

**Why `all-MiniLM-L6-v2`?**
It's fast, free, runs locally with no API calls, and performs well on short technical passages. For a student notes use case, it's the right tradeoff between quality and cost.

**Why Groq over OpenAI?**
Groq's inference is significantly faster and has a generous free tier. For a demo/portfolio project where response speed matters for user experience, Groq was the obvious choice.

---

## What's missing / future work

- Strict subject-mode filtering (currently subject mode changes prompting style, not retrieval scope)
- Better OCR post-processing — EasyOCR output on messy handwriting needs cleanup
- Persistent vector store across sessions
- Support for images and diagrams in notes

---

Built as part of a 15-day RAG from scratch project.

**Stack:** Python · LangChain · FAISS · Groq · HuggingFace · Streamlit · EasyOCR
