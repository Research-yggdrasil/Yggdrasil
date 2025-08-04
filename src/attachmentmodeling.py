# attachmentmodeling.py

from datetime import datetime
from itertools import combinations
import difflib

# Implements the Relationship Modeling module (RM), which constructs and maintains
# [cite_start]the Attachment Graph (Anne-centric) and the inter-entity Relationship Graph[cite: 87, 99].
class AuthorityAttachmentModel:
    # Maps entity aliases to a canonical name for consistent tracking.
    ENTITY_MAP = {
        "father": "Otto Frank", "daddy": "Otto Frank", "pim": "Otto Frank",
        "mother": "Edith Frank", "mommy": "Edith Frank", "mama": "Edith Frank",
        "margot": "Margot Frank", "sister": "Margot Frank",
        "peter": "Peter van Pels", "peter van daan": "Peter van Pels",
        "mr. dussel": "Fritz Pfeffer", "albert dussel": "Fritz Pfeffer",
        "mrs. van daan": "Auguste van Pels", "auguste": "Auguste van Pels",
        "mr. van daan": "Hermann van Pels", "hermann": "Hermann van Pels",
        "moortje": "Moortje", "cat": "Moortje",
        "grammy": "Grandmother Frank", "grandmother": "Grandmother Frank",
        "kitty": "Diary", "you": "Diary", "anne": "Anne Frank", "i": "Anne Frank",
        "me": "Anne Frank", "my": "Anne Frank", "bep": "Bep Voskuijl",
        "miep": "Miep Gies", "mr. kraler": "Victor Kugler", "mr. koophuis": "Johannes Kleiman",
        "the annex": "Secret Annex", "our hiding place": "Secret Annex"
    }

    # Defines the valence for each core emotion, used to calculate relationship weight adjustments.
    EMOTION_VALENCE = {"Joy": 1.0, "Love/Attachment": 1.0, "Sadness": -0.6, "Fear": -0.8, "Anger": -0.9, "Curiosity": 0.0}
    
    def __init__(self):
        self.attachment_graph = {"Anne Frank": {}}
        self.entity_graph = {}
        self.emotional_graph = {} 
        self.similarity_threshold = 0.8
        
    def normalize_entity(self, entity: str):
        entity_lower = entity.lower()
        if entity_lower in self.ENTITY_MAP:
            return self.ENTITY_MAP[entity_lower]
        best_match = max(self.ENTITY_MAP.keys(), key=lambda key: difflib.SequenceMatcher(None, entity_lower, key).ratio())
        if difflib.SequenceMatcher(None, entity_lower, best_match).ratio() >= self.similarity_threshold:
            return self.ENTITY_MAP[best_match]
        return entity.title()

    # Dynamically adjusts edge weights in the graphs based on the emotional valence
    # [cite_start]and intensity of shared experiences, as per Equation 6[cite: 135, 137].
    def update_attachment(self, entities, emotion, intensity, event_text):
        valence_multiplier = self.EMOTION_VALENCE.get(emotion, 0.0)
        weight_adjustment = intensity * valence_multiplier / 10.0
        
        normalized_entities = {self.normalize_entity(e) for e in entities} - {"Anne Frank"}
        
        for entity in normalized_entities:
            if not entity: continue
            
            if entity not in self.attachment_graph["Anne Frank"]:
                self.attachment_graph["Anne Frank"][entity] = {"weight": 0.0, "history": []}
            
            current = self.attachment_graph["Anne Frank"][entity]
            current["weight"] += weight_adjustment
            current["history"].append({"emotion": emotion, "adjustment": weight_adjustment})

    def process_event(self, event_text, entities, emotion, intensity):
        self.update_attachment(entities, emotion, intensity, event_text)

    def get_strongest_attachments(self, limit=5):
        attachments = self.attachment_graph["Anne Frank"].items()
        return sorted(attachments, key=lambda item: abs(item[1]["weight"]), reverse=True)[:limit]

    def get_attachment(self, entity):
        return self.attachment_graph["Anne Frank"].get(self.normalize_entity(entity))