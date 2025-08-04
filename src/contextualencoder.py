from openai import OpenAI
import re
import json
import spacy
from src.helper import get_response
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def extract_json(text):
    """Safely extract JSON from the model output."""
    try:
        json_text = re.search(r'\{.*\}', text, re.DOTALL).group()
        return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        print("Raw output:", text)
        return None

def process_sentence(client,sentence, event_id):
    prompt = f"""
    You are a Sensory-Event Intake system.

    Given the following text fragment, extract the fields below in JSON format.

    Rules:
    - Think like a human experiencing the moment — focus on sensory details, emotional tone, and the setting.
    - For temporal context, infer urgency and time of day if possible (e.g., "running in the dark" → Night + Urgent).
    - Sensory features should be based on perceived experiences — physical, emotional, and social cues.
    - Do not focus on abstract analysis or reasoning, only raw sensory experience and immediate emotional reaction.
    - Respond ONLY with JSON. No explanation.

    Required Fields:
    - Event ID: (given: event_{event_id})
    - Sensory Features: key descriptors (e.g., ["dark room", "cold wind", "loud footsteps", "school environment", "feeling of isolation"])
    - Temporal Context: {{"TimeOfDay": "Day" | "Night" | "Unknown","Urgency": "Urgent" | "Peaceful" | "Neutral"}}
    - Social Context: (Alone, With Family, With Strangers)
    - Raw Text: (Original sentence)

    Text:
    \"\"\"{sentence}\"\"\"

    Respond ONLY with valid JSON.
    """

    response = get_response(client,prompt)
    return extract_json(response)

def encoder(client_instance, te, event_id_start=0): # Renamed event_id to event_id_start for clarity
    sentences = split_into_sentences(te)
    events = []
    # Wrap enumerate with tqdm for a progress bar
    for idx, sentence in tqdm(enumerate(sentences, start=event_id_start), 
                              total=len(sentences), 
                              desc="Processing Sentences"):
        event = process_sentence(client_instance, sentence, idx) # Pass client_instance
        if event:
            events.append(event)
        # event_id = idx # This line is not needed if you return events and the last idx
    
    # The last event_id processed will be event_id_start + len(sentences) - 1
    # If no sentences, return event_id_start
    last_event_id = event_id_start + len(sentences) - 1 if sentences else event_id_start

    return events, last_event_id