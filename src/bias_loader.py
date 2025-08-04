import json
import random
from pathlib import Path


def load_bias_summary(bias_path="results/bias.json", contradiction_path="results/contradictionlog.json", concept=None, limit=5):
    bias_path = Path(bias_path)
    contradiction_path = Path(contradiction_path)

    bias_summary = []
    contradiction_summary = []
    ambient_bias = []
    ambient_contr = []

    if bias_path.exists():
        with open(bias_path) as f:
            bias_data = json.load(f)

        for c, emotion_map in bias_data.items():
            if len(emotion_map) > 1:
                sorted_emotions = sorted(emotion_map.items(), key=lambda x: -x[1])
                path = " → ".join(e for e, _ in sorted_emotions)
                item = f"- {c}: {path}"
                if concept and concept.lower() in c.lower():
                    bias_summary.append(item)
                else:
                    ambient_bias.append(item)

    if contradiction_path.exists():
        with open(contradiction_path) as f:
            contradiction_data = json.load(f)

        for entry in contradiction_data:
            c = entry.get("concept")
            prior = entry.get("prior_emotion")
            new = entry.get("new_emotion")
            if c and prior and new:
                item = f"- {c}: {prior} → {new}"
                if concept and concept.lower() in c.lower():
                    contradiction_summary.append(item)
                else:
                    ambient_contr.append(item)

    sampled_bias = random.sample(ambient_bias, min(2, len(ambient_bias)))
    sampled_contr = random.sample(ambient_contr, min(1, len(ambient_contr)))

    return bias_summary + sampled_bias, contradiction_summary + sampled_contr


# # Example usage:
# if __name__ == "__main__":
#     bias, contradictions = load_bias_summary(concept="Peter")
#     print("\nBias Shifts:")
#     for b in bias:
#         print(b)
#     print("\nContradictions:")
#     for c in contradictions:
#         print(c)
