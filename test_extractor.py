"""
Ru'ya Command Extractor — Test Suite
======================================
No Ollama needed to run these tests.
Run with: python test_extractor.py
"""

import json
import unittest
from unittest.mock import patch
from command_extractor import CommandExtractor, Intent


# ── Helper ────────────────────────────────────────────────────────────────────
def fake_response(intent, target=None, location_hint=None, confidence="high"):
    """Simulates a well-formed LLM JSON response."""
    return json.dumps({
        "intent": intent,
        "entities": {"target": target, "location_hint": location_hint},
        "confidence": confidence,
    })

def ext():
    return CommandExtractor(model="llama3.2:1b")


# ── ENVIRONMENT_QUERY tests ───────────────────────────────────────────────────
class TestEnvironmentQuery(unittest.TestCase):

    def test_basic(self):
        with patch.object(ext(), "_call_llm", return_value=fake_response("ENVIRONMENT_QUERY")):
            r = ext().extract("what's in front of me?")
        # Re-extract with patched instance
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("ENVIRONMENT_QUERY")):
            r = e.extract("what's in front of me?")
        self.assertEqual(r.intent, Intent.ENVIRONMENT_QUERY)

    def test_with_location(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("ENVIRONMENT_QUERY", location_hint="left")):
            r = e.extract("what's to my left?")
        self.assertEqual(r.intent, Intent.ENVIRONMENT_QUERY)
        self.assertEqual(r.entities.get("location_hint"), "left")

    def test_with_target(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("ENVIRONMENT_QUERY", target="chair")):
            r = e.extract("is there a chair nearby?")
        self.assertEqual(r.entities.get("target"), "chair")

    def test_describe_surroundings(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("ENVIRONMENT_QUERY")):
            r = e.extract("describe my surroundings")
        self.assertEqual(r.intent, Intent.ENVIRONMENT_QUERY)


# ── OCR tests ─────────────────────────────────────────────────────────────────
class TestOCR(unittest.TestCase):

    def test_read_this(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("OCR")):
            r = e.extract("read this for me")
        self.assertEqual(r.intent, Intent.OCR)

    def test_read_menu(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("OCR", target="menu")):
            r = e.extract("read the menu")
        self.assertEqual(r.intent, Intent.OCR)
        self.assertEqual(r.entities.get("target"), "menu")

    def test_what_does_sign_say(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("OCR", target="sign")):
            r = e.extract("what does the sign say?")
        self.assertEqual(r.entities.get("target"), "sign")

    def test_whats_written(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("OCR")):
            r = e.extract("what's written here?")
        self.assertEqual(r.intent, Intent.OCR)


# ── BATTERY tests ─────────────────────────────────────────────────────────────
class TestBattery(unittest.TestCase):

    def test_single_word(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("BATTERY")):
            r = e.extract("battery")
        self.assertEqual(r.intent, Intent.BATTERY)

    def test_full_phrase(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("BATTERY")):
            r = e.extract("how much battery do I have left?")
        self.assertEqual(r.intent, Intent.BATTERY)

    def test_check_power(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("BATTERY")):
            r = e.extract("check power level")
        self.assertEqual(r.intent, Intent.BATTERY)


# ── EMERGENCY tests ───────────────────────────────────────────────────────────
class TestEmergency(unittest.TestCase):

    def test_help_no_llm_call(self):
        """'help' must be caught instantly — LLM should NOT be called."""
        e = ext()
        with patch.object(e, "_call_llm") as mock_llm:
            r = e.extract("help")
        mock_llm.assert_not_called()
        self.assertEqual(r.intent, Intent.EMERGENCY)
        self.assertEqual(r.confidence, "high")

    def test_sos_no_llm_call(self):
        e = ext()
        with patch.object(e, "_call_llm") as mock_llm:
            r = e.extract("SOS")
        mock_llm.assert_not_called()
        self.assertEqual(r.intent, Intent.EMERGENCY)

    def test_i_fell_no_llm_call(self):
        e = ext()
        with patch.object(e, "_call_llm") as mock_llm:
            r = e.extract("I fell down")
        mock_llm.assert_not_called()
        self.assertEqual(r.intent, Intent.EMERGENCY)

    def test_is_emergency_helper(self):
        e = ext()
        with patch.object(e, "_call_llm") as mock_llm:
            r = e.extract("emergency!")
        self.assertTrue(r.is_emergency())

    def test_subtle_emergency_via_llm(self):
        """Subtle emergency the fast-path misses → LLM catches it."""
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("EMERGENCY")):
            r = e.extract("someone is following me")
        self.assertEqual(r.intent, Intent.EMERGENCY)


# ── Edge case tests ───────────────────────────────────────────────────────────
class TestEdgeCases(unittest.TestCase):

    def test_empty_input(self):
        e = ext()
        r = e.extract("")
        self.assertEqual(r.intent, Intent.UNKNOWN)
        self.assertIsNotNone(r.error)

    def test_whitespace_only(self):
        e = ext()
        r = e.extract("   ")
        self.assertEqual(r.intent, Intent.UNKNOWN)

    def test_unknown_intent(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("UNKNOWN")):
            r = e.extract("hello how are you")
        self.assertEqual(r.intent, Intent.UNKNOWN)

    def test_malformed_json(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value="this is not json"):
            r = e.extract("some input")
        self.assertEqual(r.intent, Intent.UNKNOWN)
        self.assertIsNotNone(r.error)

    def test_hallucinated_intent(self):
        """LLM making up an intent → falls back to UNKNOWN."""
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("DANCE")):
            r = e.extract("some input")
        self.assertEqual(r.intent, Intent.UNKNOWN)

    def test_null_entities_not_in_dict(self):
        """Null entity values should be stripped from the result."""
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("OCR")):
            r = e.extract("read this")
        self.assertNotIn("target", r.entities)
        self.assertNotIn("location_hint", r.entities)

    def test_to_dict_keys(self):
        e = ext()
        with patch.object(e, "_call_llm", return_value=fake_response("BATTERY")):
            r = e.extract("battery status")
        d = r.to_dict()
        self.assertIn("intent", d)
        self.assertIn("entities", d)
        self.assertIn("confidence", d)
        self.assertIn("raw_input", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)