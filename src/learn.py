from src.helper import compute_similarity
from typing import Dict, List, Tuple
import heapq
import uuid
from datetime import datetime 


def predict_emotion(emotional_memory_stack,new_event: Dict, k: int = 5) -> Dict:
    memory_list = emotional_memory_stack["Memory List"]
    similarity_heap: List[Tuple[float, str, Dict]] = []

    for memory in memory_list:
        similarity = compute_similarity(new_event, memory)
        if similarity > 0:
            heapq.heappush(similarity_heap, (-similarity, memory["Event ID"], memory))  # use ID to break ties

    top_k = [heapq.heappop(similarity_heap)[2] for _ in range(min(k, len(similarity_heap)))]

    if not top_k:
        return {
            "Predicted Emotion": "Neutral",
            "Predicted Intensity": 0.0,
            "Supporting Memories": []
        }

    emotion_scores = {}
    total_weight = 0.0
    supporting_ids = []

    for mem in top_k:
        emotion = mem["Assigned Emotion"]
        intensity = mem["Emotion Intensity"]
        sim = compute_similarity(new_event, mem)

        emotion_scores[emotion] = emotion_scores.get(emotion, 0) + sim * intensity
        total_weight += sim
        supporting_ids.append(mem["Event ID"])

    predicted_emotion = max(emotion_scores, key=emotion_scores.get)
    predicted_intensity = round(emotion_scores[predicted_emotion] / total_weight, 2)

    return {
        "Predicted Emotion": predicted_emotion,
        "Predicted Intensity": predicted_intensity,
        "Supporting Memories": supporting_ids
    }


def update_bias_meter(bias_meter,emotional_timeline,event):

    concept = event["Sensory Features"][0] if event.get("Sensory Features") else "unknown"
    emotion = event["Assigned Emotion"]
    intensity = event["Emotion Intensity"]
    timestamp = datetime.now().isoformat()

    if concept not in bias_meter:
        bias_meter[concept] = {}
    if emotion not in bias_meter[concept]:
        bias_meter[concept][emotion] = 0
    bias_meter[concept][emotion] += 1

    if concept not in emotional_timeline:
        emotional_timeline[concept] = []
    emotional_timeline[concept].append({
        "emotion": emotion,
        "intensity": intensity,
        "timestamp": timestamp
    })


def learn_from_emotional_error(bias_meter,emotional_timeline,contradiction_log,emotional_memory_stack,new_event: Dict, predicted: Dict, actual: Dict, error_thresholds=(0.2, 0.5)) -> Dict:
    error = abs(predicted["Predicted Intensity"] - actual["Emotion Intensity"])
    match = predicted["Predicted Emotion"] == actual["Assigned Emotion"]
    memory_list = emotional_memory_stack["Memory List"]

    similar_memories = []
    for mem in memory_list:
        sim = compute_similarity(new_event, mem)
        if sim > 0:
            similar_memories.append((sim, mem))

    similar_memories.sort(key=lambda x: -x[0])
    top_supporting = [mem for _, mem in similar_memories[:5]]

    updates = []
    added_memory = None
    contradiction_logged = False

    if error < error_thresholds[0]:  # 🔧 Small error — reinforce
        for mem in top_supporting:
            if mem["Assigned Emotion"] == predicted["Predicted Emotion"]:
                mem["Emotion Intensity"] = round(min(mem["Emotion Intensity"] + 0.05, 1.0), 2)
                updates.append(mem["Event ID"])

    elif error < error_thresholds[1]:  # 🔧 Moderate error
        if match:
            for mem in top_supporting:
                if mem["Assigned Emotion"] == predicted["Predicted Emotion"]:
                    mem["Emotion Intensity"] = round((mem["Emotion Intensity"] + actual["Emotion Intensity"]) / 2, 2)
                    updates.append(mem["Event ID"])
        else:
            contradiction_logged = True
            new_mem = {
                "Event ID": f"event_{uuid.uuid4().hex[:6]}",
                "Assigned Emotion": actual["Assigned Emotion"],
                "Emotion Intensity": round(actual["Emotion Intensity"], 2),
                "Raw Text": new_event.get("Raw Text", ""),
                "Sensory Features": new_event.get("Sensory Features", []),
                "Social Context": new_event.get("Social Context"),
                "Temporal Context": new_event.get("Temporal Context")
            }
            # emotional_memory_stack["Memory List"].append(new_mem)
            # emotional_memory_stack["Emotion Index"][actual["Emotion"]].append(new_mem["Event ID"])
            added_memory = new_mem["Event ID"]

            # 🔧 NEW: Contradiction tracking
            concept = new_event["Sensory Features"][0] if new_event.get("Sensory Features") else "unknown"
            contradiction_log.append({
                "concept": concept,
                "prior_emotion": predicted["Predicted Emotion"],
                "new_emotion": actual["Assigned Emotion"],
                "error": round(error, 2),
                "event_id": new_mem["Event ID"],
                "timestamp": datetime.now().isoformat()
            })

            # 🔧 NEW: Bias tracking
            update_bias_meter(bias_meter,emotional_timeline,new_mem)

    else:  # 🔧 High error
        if match:
            for mem in top_supporting:
                if mem["Assigned Emotion"] == predicted["Predicted Emotion"]:
                    mem["Emotion Intensity"] = round(max(mem["Emotion Intensity"] - 0.2, 0.0), 2)
                    updates.append(mem["Event ID"])
        else:
            contradiction_logged = True
            new_mem = {
                "Event ID": f"event_{uuid.uuid4().hex[:6]}",
                "Assigned Emotion": actual["Assigned Emotion"],
                "Emotion Intensity": round(actual["Emotion Intensity"], 2),
                "Raw Text": new_event.get("Raw Text", ""),
                "Sensory Features": new_event.get("Sensory Features", []),
                "Social Context": new_event.get("Social Context"),
                "Temporal Context": new_event.get("Temporal Context")
            }
            emotional_memory_stack["Memory List"].append(new_mem)
            emotional_memory_stack["Emotion Index"][actual["Assigned Emotion"]].append(new_mem["Event ID"])
            added_memory = new_mem["Event ID"]

            # 🔧 NEW: Contradiction tracking
            concept = new_event["Sensory Features"][0] if new_event.get("Sensory Features") else "unknown"
            contradiction_log.append({
                "concept": concept,
                "prior_emotion": predicted["Predicted Emotion"],
                "new_emotion": actual["Assigned Emotion"],
                "error": round(error, 2),
                "event_id": new_mem["Event ID"],
                "timestamp": datetime.now().isoformat()
            })

            # 🔧 NEW: Bias tracking
            update_bias_meter(bias_meter,emotional_timeline,new_mem)

    return {
        "Error": round(error, 2),
        "Emotion Match": match,
        "Updated Memories": updates,
        "Contradiction Logged": contradiction_logged,
        "New Memory Added": added_memory
    },bias_meter,emotional_timeline,contradiction_log


def generate_bias_shift_report(emotional_timeline,concept: str) -> Dict:
    if concept not in emotional_timeline:
        return {"error": f"No data for concept '{concept}'"}

    history = emotional_timeline[concept]
    total = len(history)
    
    emotion_counts = {}
    for entry in history:
        emotion = entry["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    emotion_distribution = {emotion: round(count / total * 100, 2) for emotion, count in emotion_counts.items()}
    dominant = max(emotion_counts.items(), key=lambda x: x[1])[0]

    # Detect shift if a newer emotion overtakes previous dominant
    recent = history[-5:]  # Last 5 events
    recent_counts = {}
    for entry in recent:
        e = entry["emotion"]
        recent_counts[e] = recent_counts.get(e, 0) + 1
    recent_dominant = max(recent_counts.items(), key=lambda x: x[1])[0]

    shift_occurred = dominant != recent_dominant

    return {
        "Concept": concept,
        "Emotion Distribution": emotion_distribution,
        "Dominant Emotion": dominant,
        "Recent Dominant": recent_dominant,
        "Shift Detected": shift_occurred,
        "Last Updated": history[-1]["timestamp"]
    }
