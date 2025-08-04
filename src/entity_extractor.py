from src.helper import get_response
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def filter_entities(entity_list):
    filtered = []
    for ent in entity_list:
        doc = nlp(ent)
        # Filter out if all tokens are stopwords, pronouns, conjunctions, etc.
        if all(token.pos_ in {"PRON", "CCONJ", "DET", "SCONJ"} or token.is_stop for token in doc):
            continue
        filtered.append(ent)
    return filtered

def extract_entities(client,text):
    prompt = f"""
    You are an entity extraction assistant for a memory-based emotional brain simulation.

    Your task is to extract only the following types of entities, exactly as they appear in the text:
    People: Names, roles, or titles (e.g., "Father", "Margot", "the policeman", "the neighbor").
    Emotionally significant objects: Tangible items with perceived importance (e.g., "diary", "ring", "letter").
    Emotionally relevant places: Specific locations or rooms mentioned (e.g., "attic", "kitchen", "hiding place").
    Emotionally charged events or actions: Specific concrete events or distinct actions that carry emotional weight (e.g., "doorbell rang", "a whispered voice", "crying", "celebrating").

    Guidelines:
    - Do not extract temporal references (e.g., "the moment when", "the time I", "quarter to seven") unless they describe a specific memorable event.
    - Do not extract phrases like "the moment I saw you" or "the moment I got you" - these are temporal references, not distinct events.
    - Do not interpret meaning, infer relationships, or add context.
    - Do not extract general items or locations unless explicitly named in the text.
    - Return results as a list of plain strings, with each string matching the exact phrasing from the text.
    - No explanations. No formatting. Only the list.

    Text:
    "{text}"

    Example output:
    ["Father", "Margot", "diary", "attic", "doorbell rang"]
    """
    response = get_response(client,prompt)
    
    # Parse the response string into a proper Python list
    try:
        # Try to safely evaluate the response as a Python expression
        import ast
        entities_list = ast.literal_eval(response.strip())
        
        # Ensure it's a list
        if not isinstance(entities_list, list):
            # If not a list, try to extract list from string
            import re
            matches = re.findall(r'\["(.+?)"\]', response.replace("'", '"'))
            if matches:
                entities_list = matches[0].split('", "')
            else:
                # Fallback: split by commas if enclosed in brackets
                cleaned = response.strip().strip('[]')
                entities_list = [item.strip().strip('"\'') for item in cleaned.split(',')]
        
        return entities_list
    except:
        # Fallback method if parsing fails
        # Remove brackets, split by commas, and clean up each item
        cleaned_response = response.strip().strip('[]')
        entities = [item.strip().strip('"\'') for item in cleaned_response.split(',')]
        return filter_entities(entities)
