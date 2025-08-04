from src.helper import get_response
import json
import re

def create_emotional_tagging_prompt(event):
    prompt = f"""
    You are a primitive Emotional Tagging System that can ONLY output in exact JSON format.
    
    ⚠️ CRITICAL INSTRUCTION ⚠️
    Your response MUST be VALID JSON with EXACTLY these fields:
    1. "Event ID": Copied directly from input
    2. "Assigned Emotion": ONE value from ["Joy", "Sadness", "Fear", "Anger", "Curiosity", "Love/Attachment"]
    3. "Emotion Intensity": Single decimal number between 0.0-1.0
    
    DO NOT include ANY explanations, reasoning, or text that is not part of the valid JSON structure.
    DO NOT add ANY additional fields or comments inside the JSON.
    
    If you're tempted to use an emotion not in the list, choose the closest match from the allowed list ONLY.
    
    INPUT EVENT:
    {json.dumps(event, indent=2)}
    
    OUTPUT FORMAT (exactly this structure with no additional text):
    {{
      "Event ID": "{event['Event ID']}",
      "Assigned Emotion": "<ONLY one of: Joy, Sadness, Fear, Anger, Curiosity, Love/Attachment>",
      "Emotion Intensity": <single decimal value between 0.0 and 1.0>
    }}
    """
    return prompt

def process_emotion_response(response_text):
    # Strip any explanatory text before or after the JSON
    json_match = re.search(r'\{[^{]*"Event ID"[^}]*\}', response_text)
    if not json_match:
        return None
    
    json_str = json_match.group(0)
    
    try:
        # Parse the JSON
        data = json.loads(json_str)
        
        # Validate Event ID exists
        if "Event ID" not in data:
            raise ValueError("Missing Event ID")
            
        # Validate emotion is in allowed list
        allowed_emotions = ["Joy", "Sadness", "Fear", "Anger", "Curiosity", "Love/Attachment"]
        if data.get("Assigned Emotion") not in allowed_emotions:
            # Replace with closest match or default
            data["Assigned Emotion"] = "Curiosity"  # Default fallback
            
        # Validate intensity is numeric and in range
        intensity = data.get("Emotion Intensity")
        if not isinstance(intensity, (int, float)) or intensity < 0 or intensity > 1:
            data["Emotion Intensity"] = 0.5  # Default fallback
            
        return data
    except json.JSONDecodeError:
        return None
    
def emotional_tagging(client,event):
    prompt = create_emotional_tagging_prompt(event)
    response = get_response(client,prompt)  # Your function to call the model
    
    # Process and validate the response
    emotion_data = process_emotion_response(response)
    
    if not emotion_data:
        # Fallback for completely invalid responses
        emotion_data = {
            "Event ID": event["Event ID"],
            "Assigned Emotion": "Curiosity",  # Default emotion
            "Emotion Intensity": 0.5  # Default intensity
        }
    
    return emotion_data