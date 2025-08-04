from openai import OpenAI
from typing import Dict
from datetime import datetime
import re

def llm() -> OpenAI:
    """
    Initializes and returns an OpenAI client instance configured for a specific API endpoint.

    This function centralizes the client configuration, making it easy to manage
    API credentials and model endpoints across the application. It encapsulates the
    setup details for connecting to the NVIDIA API endpoint.

    Returns:
        OpenAI: An authenticated client object ready to make API calls.
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="NVIDIA_NIM_API_KEY" # Replace with your actual API key.
    )
    return client

def get_response(client: OpenAI, prompt: str) -> str:
    """
    Sends a prompt to the specified LLM and returns the content of its response.

    This is a wrapper function for the chat completions API call, standardizing the
    model parameters (e.g., model name, temperature, top_p) for consistent
    behavior.

    Args:
        client (OpenAI): The initialized API client.
        prompt (str): The user prompt to send to the model.

    Returns:
        str: The textual content of the model's message.
    """
    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, # Lower temperature for more deterministic, less creative output.
        top_p=0.7,       # Nucleus sampling to control diversity.
        max_tokens=4096
    )
    return response.choices[0].message.content

def compute_similarity(event_a: Dict, event_b: Dict) -> float:
    """
    Calculates a similarity score between two structured event dictionaries.

    The similarity is computed as a weighted sum of overlaps in their features:
    - Sensory Features: Uses the Jaccard index (intersection over union) for set-based comparison.
    - Social Context: A binary score for an exact match.
    - Temporal Context: A binary score for an exact match.

    Args:
        event_a (Dict): The first event dictionary.
        event_b (Dict): The second event dictionary.

    Returns:
        float: A similarity score, capped at 1.0.
    """
    score = 0.0

    # Compare Sensory Features using Jaccard Similarity.
    if event_a.get("Sensory Features") and event_b.get("Sensory Features"):
        set_a = set(event_a["Sensory Features"])
        set_b = set(event_b["Sensory Features"])
        overlap = set_a.intersection(set_b)
        union = set_a.union(set_b)
        score += len(overlap) / max(len(union), 1)

    # Add a fixed bonus for matching Social Context.
    if event_a.get("Social Context") and event_a["Social Context"] == event_b.get("Social Context"):
        score += 0.5

    # Add a fixed bonus for matching Temporal Context.
    if event_a.get("Temporal Context") and event_a["Temporal Context"] == event_b.get("Temporal Context"):
        score += 0.5

    # Cap the total score at 1.0 to maintain a normalized range.
    return min(score, 1.0)

def compute_dominant_emotion(emotional_memory_stack: Dict) -> str:
    """
    Determines the overall dominant emotion in the memory stack.

    This is calculated by summing the intensity of all memories for each emotion
    category and identifying the category with the highest total score. It represents
    the model's current "mood" or emotional disposition.

    Args:
        emotional_memory_stack (Dict): The main memory data structure.

    Returns:
        str: The name of the dominant emotion (e.g., "Joy"), or "Neutral" if empty.
    """
    emotion_totals = {}
    for mem in emotional_memory_stack["Memory List"]:
        emo = mem["Assigned Emotion"]
        intensity = mem["Emotion Intensity"]
        emotion_totals[emo] = emotion_totals.get(emo, 0) + intensity

    if not emotion_totals:
        return "Neutral"

    # Find the emotion with the maximum aggregated intensity.
    return max(emotion_totals.items(), key=lambda x: x[1])[0]

def extract_entry_date(entry: str) -> str:
    """
    Parses a date string from the beginning of a diary entry.

    Args:
        entry (str): The full entry string, expected to start with a date.

    Returns:
        str: The date in "YYYY-MM-DD" format, or "Unknown" on failure.
    """
    try:
        # Assumes date is the first part of a comma-separated string.
        parts = entry.split(',', 3)
        date_str = ','.join(parts[:3]).strip()
        # Parse the string into a datetime object and then reformat it.
        date_obj = datetime.strptime(date_str, "%A, %B %d, %Y")
        return date_obj.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"[⚠️] Failed to extract date from entry: {e}")
        return "Unknown"

def extract_clean_emotion(text: str) -> str:
    """
    Extracts a canonical emotion name from a string that may contain other text.

    Uses a regular expression to find one of the predefined emotion keywords within
    the input text, ignoring case. This is useful for cleaning up potentially messy
    LLM outputs.

    Args:
        text (str): The string to search within.

    Returns:
        str: The canonical emotion name (e.g., "Love/Attachment") or "Unknown" if not found.
    """
    # Regex with word boundaries (\b) ensures "Anger" matches but "dangerous" does not.
    match = re.search(r'\b(Joy|Sadness|Fear|Anger|Curiosity|Love/Attachment)\b', text, re.IGNORECASE)
    # Return the found emotion in title case for consistency.
    return match.group(0).title() if match else "Unknown"