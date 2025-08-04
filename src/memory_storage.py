def store_memory(emotional_memory_stack,event, emotion_tag):
    """Combines sensory event and emotional tag into a structured memory unit."""
    memory_unit = {
        "Event ID": event["Event ID"],
        "Sensory Features": event["Sensory Features"],
        "Temporal Context": event["Temporal Context"],
        "Social Context": event["Social Context"],
        "Raw Text": event["Raw Text"],
        "Assigned Emotion": emotion_tag["Assigned Emotion"],
        "Emotion Intensity": emotion_tag["Emotion Intensity"]
    }
    
    # Append to memory list (chronological order)
    emotional_memory_stack["Memory List"].append(memory_unit)
    
    # Update Emotion Index
    assigned_emotion = emotion_tag["Assigned Emotion"]
    if assigned_emotion in emotional_memory_stack["Emotion Index"]:
        emotional_memory_stack["Emotion Index"][assigned_emotion].append(event["Event ID"])
    else:
        emotional_memory_stack["Emotion Index"][assigned_emotion] = [event["Event ID"]]
