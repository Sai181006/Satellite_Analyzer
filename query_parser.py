import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
try:
    gemini = genai.GenerativeModel("gemini-1.5-flash")
    GEMINI_AVAILABLE = True
except Exception as e:
    print(f"[Gemini init failed] {e}")
    gemini = None
    GEMINI_AVAILABLE = False

# Map unsupported terms to YOLO-detectable equivalents
TERM_MAP = {
    "building": "car",       # infer urban density from vehicles
    "buildings": "car",
    "road": "car",
    "roads": "car",
    "structure": "truck",
    "structures": "truck",
}

GEMINI_PROMPT = """
You are a query parser for a satellite image analysis system.
The system can only detect these object classes: car, truck, bus, person, airplane, boat, motorcycle.
If the user mentions buildings, roads, or structures — map them to vehicle classes (car or truck).

Convert the user query into a JSON object. Return ONLY raw JSON, no markdown, no explanation.

Format:
{
  "object": "<one of: car, truck, bus, person, airplane, boat, motorcycle>",
  "condition": "<one of: all, high_density, low_density, near>",
  "relation": "<one of: none, near, inside>",
  "target": "<reference object class or null>"
}

User query: {query}
"""

def parse_query(query: str) -> tuple:
    """Try Gemini first, fall back to rule-based parser. Returns (parsed_dict, source)."""
    if not GEMINI_AVAILABLE:
        return (_fallback_parser(query), "fallback")
    try:
        prompt = GEMINI_PROMPT.format(query=query)
        response = gemini.generate_content(prompt)
        text = response.text.strip()
        text = re.sub(r"```json|```", "", text).strip()
        parsed = json.loads(text)
        # Remap unsupported terms
        parsed["object"] = TERM_MAP.get(parsed.get("object", ""), parsed.get("object", "car"))
        if parsed.get("target"):
            parsed["target"] = TERM_MAP.get(parsed["target"], parsed["target"])
        return (parsed, "gemini")
    except Exception as e:
        print(f"[Gemini failed] {e} — using fallback parser")
        return (_fallback_parser(query), "fallback")

def _fallback_parser(query: str) -> dict:
    """Rule-based keyword parser."""
    q = query.lower()

    OBJECT_KEYWORDS = ["car", "truck", "bus", "person", "airplane", "boat", "motorcycle"]
    CONDITION_MAP = {
        "dense": "high_density", "many": "high_density", "lot": "high_density",
        "crowded": "high_density", "few": "low_density", "sparse": "low_density",
        "near": "near", "close": "near", "next": "near"
    }

    # Remap unsupported terms before matching
    for term, replacement in TERM_MAP.items():
        q = q.replace(term, replacement)

    detected_obj = next((w for w in OBJECT_KEYWORDS if w in q), "car")
    condition = next((v for k, v in CONDITION_MAP.items() if k in q), "all")
    relation = "near" if "near" in q or "close" in q else "none"
    target = None
    if relation == "near":
        target = next((w for w in OBJECT_KEYWORDS if w in q and w != detected_obj), None)

    return {
        "object": detected_obj,
        "condition": condition,
        "relation": relation,
        "target": target
    }
