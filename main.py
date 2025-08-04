# main.py

# ======================================================================================
# Yggdrasil Agent - Main Execution Script
# 
# Orchestrates the two-phase learning process as outlined in the paper.
# Phase 1 (Model Seeding) builds the foundational memory, while Phase 2
# (Contradiction-Driven Learning) actively adapts the agent's internal model.
# ======================================================================================

from src.entries import readandmakeentries
from src.helper import llm, compute_dominant_emotion, extract_entry_date, extract_clean_emotion
from src.contextualencoder import encoder
from src.emotionaltagger import emotional_tagging
from src.memory_storage import store_memory
from src.entity_extractor import extract_entities
from src.learn import predict_emotion, learn_from_emotional_error, generate_bias_shift_report
from src.attachmentmodeling import AuthorityAttachmentModel
from tqdm import tqdm
import json

# --- Global logs for tracking agent's learning and internal state.
contradiction_log = []
bias_meter = {}
emotional_timeline = {}

# --- Initialize the agent's core emotional memory stack (M).
# The agent begins with no predefined world model.
emotional_memory_stack = {
    "Memory List": [],
    "Emotion Index": {
        "Joy": [],
        "Sadness": [],
        "Fear": [],
        "Anger": [],
        "Curiosity": [],
        "Love/Attachment": []
    }
}

# --- Initialize the Relationship Modeling module (RM).
attachment_model = AuthorityAttachmentModel()

# --- Load and partition the dataset from Anne Frank's diary.
entries = readandmakeentries("data\\the-diary-of-anne-frank.pdf")
phase1entries = entries[1:8]
phase2entries = entries[8:13]

client = llm()
eventid = 0

print("=" * 60)
print("PHASE 1: BUILDING INITIAL EMOTIONAL MODEL (MODEL SEEDING)")
print("=" * 60)

# ======================================================================================
# Phase 1: Model Seeding
# Populates the agent's memory systems without active learning, establishing
# foundational emotional and social representations.
# ======================================================================================
for entry_idx, entry in enumerate(phase1entries):
    print(f"\nProcessing Phase 1 Entry {entry_idx + 1}/{len(phase1entries)}")
    
    # Event Formalization via the Perception Layer.
    events, eventid = encoder(client, entry, eventid + 1)   
    
    for event in tqdm(events, desc=f"Processing Entry {entry_idx + 1} events"):
        # Affective Grounding (High Road) assigns a ground-truth emotional tag.
        e_tag = emotional_tagging(client, event)
        
        if e_tag is None:
            continue
            
        # Store the emotionally tagged event in the memory stack (M).
        store_memory(emotional_memory_stack, event, e_tag)
        
        # Initial Social Modeling extracts entities to build relationship graphs.
        raw_text = event.get('Raw Text', '')
        if raw_text:
            entities = extract_entities(client, raw_text)
            if entities:
                attachment_model.process_event(
                    raw_text, 
                    entities, 
                    e_tag['Assigned Emotion'], 
                    e_tag['Emotion Intensity']
                )

print(f"\nPhase 1 Complete!")
print(f"Total memories stored: {len(emotional_memory_stack['Memory List'])}")
strongest_attachments = attachment_model.get_strongest_attachments(5)
print(f"Strongest attachments after Phase 1: {strongest_attachments}")

print("\n" + "=" * 60)
print("PHASE 2: CONTRADICTION-DRIVEN LEARNING AND ADAPTATION")
print("=" * 60)

# ======================================================================================
# Phase 2: Contradiction-Driven Learning
# Engages the full dual-pathway learning cycle, where the agent predicts,
# compares, and adapts based on emotional contradictions.
# ======================================================================================
phase2_stats = {
    "total_events": 0, "predictions_made": 0, "contradictions": 0,
    "new_memories_added": 0, "prediction_errors": [], "shifted_concepts": set()
}

for entry_idx, entry in enumerate(phase2entries):
    print(f"\nProcessing Phase 2 Entry {entry_idx + 1}/{len(phase2entries)}")
    
    events, eventid = encoder(client, entry, eventid + 1)
    
    for event in tqdm(events, desc=f"Learning from Entry {entry_idx + 1} events"):
        phase2_stats["total_events"] += 1
        
        # Low Road: Predicts emotional content based on accumulated memory.
        predicted = predict_emotion(emotional_memory_stack, event)
        phase2_stats["predictions_made"] += 1
        
        # High Road: Obtains the ground-truth emotional tag for the event.
        actual = emotional_tagging(client, event)
        
        if actual is None:
            continue
        
        cleaned_emotion = extract_clean_emotion(actual.get("Assigned Emotion", ""))
        if cleaned_emotion == "Unknown":
            continue
        actual["Assigned Emotion"] = cleaned_emotion
        
        print(f"\nEvent ID: {event.get('Event ID', 'unknown')}")
        print(f"Predicted: {predicted['Predicted Emotion']} ({predicted['Predicted Intensity']:.2f})")
        print(f"Actual: {actual['Assigned Emotion']} ({actual['Emotion Intensity']:.2f})")
        
        # Core learning step where prediction error drives memory adaptation.
        learning_result,bias_meter,emotional_timeline,contradiction_log = learn_from_emotional_error(
            bias_meter, emotional_timeline, contradiction_log, emotional_memory_stack,
            new_event=event, predicted=predicted, actual=actual
        )
        
        print(f"Learning result: Error={learning_result['Error']:.2f}, Match={learning_result['Emotion Match']}")
        
        phase2_stats["prediction_errors"].append(learning_result["Error"])
        if learning_result["Contradiction Logged"]:
            phase2_stats["contradictions"] += 1
        if learning_result["New Memory Added"]:
            phase2_stats["new_memories_added"] += 1
            store_memory(emotional_memory_stack, event, actual)
            
            concept = event["Sensory Features"][0] if event.get("Sensory Features") else "unknown"
            report = generate_bias_shift_report(emotional_timeline,concept)
            if report.get("Shift Detected", False):
                phase2_stats["shifted_concepts"].add(concept)
        
        # Dynamically updates social model based on the event's emotional tone.
        raw_text = event.get('Raw Text', '')
        if raw_text:
            entities = extract_entities(client, raw_text)
            if entities:
                attachment_model.process_event(
                    raw_text, entities, actual['Assigned Emotion'], actual['Emotion Intensity']
                )

print("\n" + "=" * 60)
print("PHASE 2 COMPLETE - LEARNING STATISTICS")
print("=" * 60)

avg_error = sum(phase2_stats["prediction_errors"]) / len(phase2_stats["prediction_errors"]) if phase2_stats["prediction_errors"] else 0.0

print(f"Total events processed: {phase2_stats['total_events']}")
print(f"Contradictions found: {phase2_stats['contradictions']}")
print(f"Average prediction error: {avg_error:.3f}")
print(f"Concepts with emotional shifts: {list(phase2_stats['shifted_concepts'])}")

print(f"\nFinal memory count: {len(emotional_memory_stack['Memory List'])}")
final_attachments = attachment_model.get_strongest_attachments(10)
print("Final strongest attachments:")
for entity, weight in final_attachments:
    attachment_data = attachment_model.get_attachment(entity)
    valence = attachment_data.get('valence', 'Unknown') if attachment_data else 'Unknown'
    print(f"  {entity}: {weight:.3f} ({valence})")

print("\nSaving results...")
with open("results/emotional_memory_stack.json", "w") as f:
    json.dump(emotional_memory_stack, f, indent=2, default=str)
with open("results/attachment_graphs.json", "w") as f:
    json.dump({"attachment_graph": attachment_model.attachment_graph, "entity_graph": attachment_model.entity_graph, "emotional_graph": attachment_model.emotional_graph}, f, indent=2, default=str)
with open("results/learning_stats.json", "w") as f:
    json.dump({**phase2_stats, "shifted_concepts": list(phase2_stats["shifted_concepts"]), "average_error": avg_error}, f, indent=2, default=str)
with open("results/bias.json", "w") as f:
    json.dump(bias_meter, f, indent=2, default=str)
with open("results/emotional_time.json", "w") as f:
    json.dump(emotional_timeline, f, indent=2, default=str)
with open("results/contradictionlog.json", "w") as f:
    json.dump(contradiction_log, f, indent=2, default=str)

print("Results saved to results/ directory.")