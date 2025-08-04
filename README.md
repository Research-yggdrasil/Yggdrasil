# Project Midgard: Emotional Memory Simulation

A Python-based system that processes diary entries, extracts sensory events, tags them with emotions, builds an emotional memory model, learns through contradiction-driven updates, and models attachment graphs.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [Module Details](#module-details)
* [Output & Results](#output--results)

---

## Overview

This project implements an emotionally grounded cognitive architecture inspired by the Yggdrasil system. It reads diary entries (e.g., Anne Frank’s diary), splits them into sensory events, tags each event with an emotion, stores structured memories, predicts emotions for new events, learns from prediction errors, and models attachment relationships among entities.

## Features

* **Event Extraction**: Splits text into sentences and extracts sensory features, temporal and social context.
* **Emotional Tagging**: Uses LLM prompts to assign discrete emotions and intensities.
* **Memory Storage**: Builds an indexed emotional memory stack.
* **Emotion Prediction & Learning**: Predicts emotions based on past memories and updates via contradiction-driven reinforcement.
* **Attachment Modeling**: Builds graphs of relationships between Anne Frank and entities, capturing emotional valence.
* **Bias & Contradiction Tracking**: Records prediction errors, logs contradictions, and tracks bias shifts.

## Repository Structure

```
├── data/
│   └── the-diary-of-anne-frank.pdf   # Input PDF for diary entries
├── results/                          # Generated output JSON files
├── src/
│   ├── entries.py                   # PDF parsing and entry segmentation
│   ├── helper.py                    # LLM client, similarity & utility functions
│   ├── contextualencoder.py         # Sentence-level event extraction
│   ├── emotionaltagger.py           # JSON-based emotion tagging
│   ├── memory_storage.py            # Store and index memories
│   ├── entity_extractor.py          # SpaCy + LLM entity extraction
│   ├── learn.py                     # Emotion prediction & learning logic
│   ├── attachmentmodeling.py        # AuthorityAttachmentModel class
│   └── main.py                      # Orchestrates Phase 1 & Phase 2 workflows
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/project-midgard.git
   cd project-midgard
   ```
2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows use `venv\\Scripts\\activate`
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
4. **Download spaCy model**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

## Configuration

1. **API Credentials**: Edit `src/helper.py` and set your LLM endpoint and API key:

   ```python
   client = OpenAI(
       base_url="https://integrate.api.nvidia.com/v1",
       api_key="<YOUR_API_KEY>"
   )
   ```
2. **Input Data**: Diary is taken from https://mrparratore.weebly.com/uploads/1/1/0/0/110095453/anne_frank_-_the_diary_of_a_young_girl_book_website.pdf and is available in the data folder

## Usage

Run the main workflow:

```bash
python src/main.py
```

* **Phase 1**: Builds initial emotional memory from the first set of entries.
* **Phase 2**: Predicts emotions, learns from errors, and updates memories & attachments.

Progress and summaries are printed to the console. Upon completion, results are saved in `results/`:

* `emotional_memory_stack.json`
* `attachment_graphs.json`
* `learning_stats.json`
* `bias.json`
* `emotional_time.json`
* `contradictionlog.json`

## Module Details

* **entries.py**: Splits PDF text into dated entries.
* **contextualencoder.py**: Uses spaCy for sentence splitting and LLM prompts to extract sensory event JSON.
* **emotionaltagger.py**: Prompts an LLM to assign an emotion and intensity to each event.
* **memory\_storage.py**: Stores events in an emotion-indexed memory stack.
* **entity\_extractor.py**: Extracts relevant entities via spaCy filtering and LLM assistance.
* **learn.py**: Implements k‑nearest memory retrieval for emotion prediction, contradiction detection, bias updates, and learning rules.
* **attachmentmodeling.py**: Defines an authority attachment graph, updating relationship weights based on emotional interactions.
* **helper.py**: Central utilities including LLM client setup, similarity calculations, date extraction, and cleaning.
* **main.py**: Coordinates the end-to-end simulation phases.



## License

This project is licensed under the MIT License. See `LICENSE` for details.
