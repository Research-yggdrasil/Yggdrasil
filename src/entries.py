import fitz
import re

def readandmakeentries(pdf_path):
    doc = fitz.open(pdf_path)

    date_pattern = re.compile(
        r"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s*)?"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"\d{1,2},\s+194[2-4]", re.IGNORECASE
    )

    entries = []
    current_entry = {"date": None, "text": ""}

    for page in doc:
        text = page.get_text()
        lines = text.split('\n')
        for line in lines:
            match = date_pattern.match(line.strip())
            if match:
                if current_entry["date"] and current_entry["text"].strip():
                    entries.append(current_entry)
                current_entry = {"date": line.strip(), "text": ""}
            elif current_entry["date"]:
                current_entry["text"] += line.strip() + " "

    if current_entry["date"] and current_entry["text"].strip():
        entries.append(current_entry)
    segmented = []
    for i, entry in enumerate(entries):
        segmented.append(f"{entry['date']} , {entry['text']}")
    return segmented
