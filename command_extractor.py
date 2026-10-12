"""
Ru'ya Smart Glasses — LLM Command Extractor
============================================
Converts ASR-transcribed speech into structured commands
using a local Ollama LLM (fully offline, no API key needed).

Processing order:
  1. Empty input          → UNKNOWN (no LLM)
  2. Emergency keywords   → EMERGENCY (no LLM)
  3. Battery keywords     → BATTERY (no LLM)
  4. OCR keywords         → OCR (no LLM)
  5. Environment keywords → ENVIRONMENT_QUERY (no LLM)
  6. Anything else        → LLM
"""

import json
import re
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

import ollama

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ruya.command_extractor")


# ── Intents ───────────────────────────────────────────────────────────────────
class Intent(str, Enum):
    ENVIRONMENT_QUERY = "ENVIRONMENT_QUERY"  # What's around/in front of me
    OCR               = "OCR"               # Read text / signs / labels
    BATTERY           = "BATTERY"           # Battery / power status
    EMERGENCY         = "EMERGENCY"         # SOS / help / danger
    UNKNOWN           = "UNKNOWN"           # Fallback


# ── Output ────────────────────────────────────────────────────────────────────
@dataclass
class CommandResult:
    intent:     Intent
    entities:   dict          = field(default_factory=dict)
    confidence: str           = "high"
    raw_input:  str           = ""
    error:      Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "intent":     self.intent.value,
            "entities":   self.entities,
            "confidence": self.confidence,
            "raw_input":  self.raw_input,
            "error":      self.error,
        }

    def is_emergency(self) -> bool:
        return self.intent == Intent.EMERGENCY

    def __str__(self) -> str:
        ent = f"  entities   : {self.entities}\n" if self.entities else ""
        err = f"  error      : {self.error}\n"    if self.error    else ""
        return (
            f"\nCommandResult\n"
            f"  intent     : {self.intent.value}\n"
            f"{ent}"
            f"  confidence : {self.confidence}\n"
            f"  raw_input  : \"{self.raw_input}\"\n"
            f"{err}"
        )


# ── System Prompt (LLM fallback only) ────────────────────────────────────────
SYSTEM_PROMPT = """
You are an intent classifier and entity extractor for smart glasses used by a visually impaired user.

Your task:
Read the user input and return a JSON object describing the detected intent and entities.

--------------------------------
RULE 1 — CHOOSE EXACTLY ONE INTENT
--------------------------------

BATTERY
Input asks about battery level or remaining usage time.
Examples:
battery
battery percent
battery level
charge
power left
how much battery
how long will it last
how long do i have left
remaining battery time

EMERGENCY
User asks for help or indicates danger.
Examples:
help
sos
call help
i fell
i'm hurt
ambulance
danger
emergency

OCR
User wants text to be read.
Examples:
read
scan
what does it say
what's written
read the text
read the sign
read this
scan the label
read the menu

ENVIRONMENT_QUERY
User asks about objects or surroundings.
Examples:
what's around me
what is in front of me
what's behind me
what's to my left
what's to my right
is there a chair
is there something ahead
describe my surroundings
what objects are here
is there an obstacle ahead

UNKNOWN
Anything else.
Examples:
hi
hello
good morning
tell me a joke
what time is it
weather

--------------------------------
RULE 2 — ENTITY EXTRACTION
--------------------------------

Return entities inside:

"entities": {
  "target": string or null,
  "location_hint": string or null
}

TARGET
Extract only a specific physical object if mentioned.

Examples of valid targets:
menu, sign, label, board, door, chair, table, person, car, stairs, wall, obstacle

Rules:
- Must be a physical object.
- Never a direction.
- Never "battery".
- Never "time".
- If no object mentioned → null.

LOCATION_HINT
If the user mentions a direction, normalize it to one of these values:
front, behind, left, right, above, below, ahead

Recognize variations and normalize:
"in front", "infront", "front of me" → front
"behind me", "at the back" → behind
"to my left", "on my left side" → left
"to my right", "on my right side" → right
"ahead", "in front ahead" → ahead
"above me", "up there" → above
"below me", "down there" → below

If no direction is mentioned → null.

--------------------------------
RULE 3 — CONFIDENCE
--------------------------------

high   → clear intent
medium → likely but vague
low    → unclear

--------------------------------
RULE 4 — OUTPUT FORMAT
--------------------------------

Return ONLY raw JSON. No markdown. No explanation. No extra text.
All null values must be real JSON null, never the string "null".

{
  "intent": "...",
  "entities": {
    "target": "...",
    "location_hint": "..."
  },
  "confidence": "..."
}

--------------------------------
EXAMPLES
--------------------------------

input: "hi"
{"intent":"UNKNOWN","entities":{"target":null,"location_hint":null},"confidence":"high"}

input: "battery"
{"intent":"BATTERY","entities":{"target":null,"location_hint":null},"confidence":"high"}

input: "how long do i have left"
{"intent":"BATTERY","entities":{"target":null,"location_hint":null},"confidence":"high"}

input: "read the menu"
{"intent":"OCR","entities":{"target":"menu","location_hint":null},"confidence":"high"}

input: "read the sign in front of me"
{"intent":"OCR","entities":{"target":"sign","location_hint":"front"},"confidence":"high"}

input: "can u read the text infront of me"
{"intent":"OCR","entities":{"target":null,"location_hint":"front"},"confidence":"high"}

input: "what's in front of me"
{"intent":"ENVIRONMENT_QUERY","entities":{"target":null,"location_hint":"front"},"confidence":"high"}

input: "is there a chair to my left"
{"intent":"ENVIRONMENT_QUERY","entities":{"target":"chair","location_hint":"left"},"confidence":"high"}

input: "describe my surroundings"
{"intent":"ENVIRONMENT_QUERY","entities":{"target":null,"location_hint":null},"confidence":"high"}

input: "help"
{"intent":"EMERGENCY","entities":{"target":null,"location_hint":null},"confidence":"high"}

input: "good morning"
{"intent":"UNKNOWN","entities":{"target":null,"location_hint":null},"confidence":"high"}

CRITICAL: direction words (front, behind, left, right, ahead, above, below, infront, back)
are ALWAYS location_hint and NEVER target.
""".strip()


# ── Extractor ─────────────────────────────────────────────────────────────────
class CommandExtractor:
    """
    Converts raw ASR text into a structured CommandResult.
    Rule-based fast-paths handle most intents instantly.
    LLM is only called for ambiguous or unknown inputs.
    """

    # ── Keyword sets ──────────────────────────────────────────────────────────

    EMERGENCY_KEYWORDS = {
        "help", "sos", "emergency", "mayday", "danger",
        "i fell", "i'm hurt", "i am hurt", "call ambulance",
        "call for help", "in danger", "need help", "save me",
        "injured", "im injured", "i'm injured", "i am injured",
        "accident", "bleeding",
    }

    BATTERY_KEYWORDS = {
        "battery", "percent", "percentage", "charge", "charging",
        "power", "how long do i have", "how much time do i have",
        "running out", "time left", "how long left", "remaining",
        "battery low", "is it charged", "how charged", "how much battery",
        "battery level", "battery status", "power left", "how long will it last",
    }

    OCR_KEYWORDS = {
        "read", "what does it say", "what's written",
        "what is written", "scan", "ocr",
        "can you read", "can u read", 
        "what does", "what do", "says", " say",
    }

    ENVIRONMENT_KEYWORDS = {
        "what's around", "what is around", "describe",
        "surroundings", "what's in front", "what is in front",
        "what's to my left", "what's to my right",
        "what's behind", "what is behind", "what's above",
        "what's below", "is there", "are there", "any obstacle",
        "anything ahead", "anything around", "who is near",
        "what's near", "what is near", "can i pass", "is it clear",
        "infront", "in front", "something in", "anything in",
        "is something", "is anything", "what is to my",
        "what's to my", "what objects", "obstacle ahead", "ahead of me", "path ahead", "the way ahead",
    }

    # Direction normalization — longer phrases checked first
    DIRECTION_MAP = {
        "in front":    "front",
        "infront":     "front",
        "at the back": "behind",
        "to my left":  "left",
        "on my left":  "left",
        "to my right": "right",
        "on my right": "right",
        "up there":    "above",
        "down there":  "below",
        "front":       "front",
        "behind":      "behind",
        "back":        "behind",
        "left":        "left",
        "right":       "right",
        "ahead":       "ahead",
        "above":       "above",
        "below":       "below",
    }

    # Words never valid as an OCR target
    OCR_TARGET_FILLER = {
        "this", "that", "it", "me", "out", "aloud", "text",
        "infront", "front", "left", "right", "behind",
        "ahead", "above", "below", "back",
    }

    # Pattern to extract OCR target: "read the menu" → "menu"
    OCR_TARGET_PATTERN = re.compile(
        r"\bread\s+(?:the\s+)?(\w+)", re.IGNORECASE
    )

    # ─────────────────────────────────────────────────────────────────────────
    #llama3.2:1b
    #qwen2.5:1.5b
    def __init__(self, model: str = "llama3.2:1b", max_retries: int = 2):
        self.model = model
        self.max_retries = max_retries

    # ── Public entry point ────────────────────────────────────────────────────

    def extract(self, transcribed_text: str) -> CommandResult:
        """Pass in raw ASR text → get back a CommandResult."""
        text  = transcribed_text.strip()
        lower = text.lower()

        # 1. Empty input
        if not text:
            return CommandResult(
                intent=Intent.UNKNOWN,
                raw_input=text,
                error="Empty input from ASR",
            )

        # 2. Emergency fast-path
        if self._is_emergency(lower):
            logger.warning("EMERGENCY fast-path: '%s'", text)
            return CommandResult(
                intent=Intent.EMERGENCY,
                entities={},
                confidence="high",
                raw_input=text,
            )

        # 3. Battery fast-path
        if self._is_battery(lower):
            logger.info("BATTERY fast-path: '%s'", text)
            return CommandResult(
                intent=Intent.BATTERY,
                entities={},
                confidence="high",
                raw_input=text,
            )

        # 4. OCR fast-path
        ocr = self._try_ocr(lower, text)
        if ocr:
            logger.info("OCR fast-path: '%s'", text)
            return ocr

        # 5. Environment fast-path
        env = self._try_environment(lower, text)
        if env:
            logger.info("ENVIRONMENT fast-path: '%s'", text)
            return env

        # 6. LLM fallback
        logger.info("LLM fallback: '%s'", text)
        return self._llm_extract(text)

    # ── Rule-based helpers ────────────────────────────────────────────────────

    def _is_emergency(self, lower: str) -> bool:
        return any(kw in lower for kw in self.EMERGENCY_KEYWORDS)

    def _is_battery(self, lower: str) -> bool:
        return any(kw in lower for kw in self.BATTERY_KEYWORDS)

    def _extract_direction(self, lower: str) -> Optional[str]:
        """Return normalized direction from input, or None.
        Longer phrases are checked first to avoid partial matches."""
        for phrase in sorted(self.DIRECTION_MAP, key=len, reverse=True):
            if phrase in lower:
                return self.DIRECTION_MAP[phrase]
        return None

    def _try_ocr(self, lower: str, original: str) -> Optional[CommandResult]:
        if not any(kw in lower for kw in self.OCR_KEYWORDS):
            return None

        entities = {}

        # Extract target object: "read the menu" → "menu"
        match = self.OCR_TARGET_PATTERN.search(original)
        if match:
            candidate = match.group(1).lower()
            if candidate not in self.DIRECTION_MAP and candidate not in self.OCR_TARGET_FILLER:
                entities["target"] = candidate

        # Extract direction if present: "read the sign in front of me"
        direction = self._extract_direction(lower)
        if direction:
            entities["location_hint"] = direction

        return CommandResult(
            intent=Intent.OCR,
            entities=entities,
            confidence="high",
            raw_input=original,
        )

    def _try_environment(self, lower: str, original: str) -> Optional[CommandResult]:
        if not any(kw in lower for kw in self.ENVIRONMENT_KEYWORDS):
            return None

        entities = {}

        direction = self._extract_direction(lower)
        if direction:
            entities["location_hint"] = direction

        return CommandResult(
            intent=Intent.ENVIRONMENT_QUERY,
            entities=entities,
            confidence="high",
            raw_input=original,
        )

    # ── LLM fallback ──────────────────────────────────────────────────────────

    def _llm_extract(self, text: str) -> CommandResult:
        for attempt in range(1, self.max_retries + 2):
            try:
                raw_json = self._call_llm(text)
                result   = self._parse_response(raw_json, text)
                logger.info("LLM → Intent: %s | Confidence: %s", result.intent.value, result.confidence)
                return result

            except _RetryableError as e:
                if attempt <= self.max_retries:
                    wait = attempt * 1.0
                    logger.warning("Attempt %d failed. Retrying in %.1fs...", attempt, wait)
                    time.sleep(wait)
                else:
                    return CommandResult(
                        intent=Intent.UNKNOWN,
                        raw_input=text,
                        error=f"LLM error after {self.max_retries + 1} attempts: {e}",
                    )

            except Exception as e:
                return CommandResult(
                    intent=Intent.UNKNOWN,
                    raw_input=text,
                    error=str(e),
                )

    def _call_llm(self, text: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                format="json",
                options={
                    "temperature": 0,
                    "num_predict": 100,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            raise _RetryableError(str(e))

    def _parse_response(self, raw: str, original_input: str) -> CommandResult:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return CommandResult(
                intent=Intent.UNKNOWN,
                raw_input=original_input,
                error="LLM returned no valid JSON",
            )

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return CommandResult(
                intent=Intent.UNKNOWN,
                raw_input=original_input,
                error="LLM returned malformed JSON",
            )

        intent_str = data.get("intent", "UNKNOWN").upper()
        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.UNKNOWN

        raw_entities = data.get("entities") or {}
        entities = {
            k: v for k, v in {
                "target":        raw_entities.get("target"),
                "location_hint": raw_entities.get("location_hint"),
            }.items() if v is not None
        }

        confidence = data.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        return CommandResult(
            intent=intent,
            entities=entities,
            confidence=confidence,
            raw_input=original_input,
        )


class _RetryableError(Exception):
    pass
