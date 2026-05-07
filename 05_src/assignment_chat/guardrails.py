"""Pure-Python pre/post filters for restricted topics and prompt-injection attempts.

The model must not discuss cats/dogs, horoscopes/zodiac, or Taylor Swift, and the
system prompt must remain hidden and unchangeable. We defend in three layers:

1. ``check_user_input`` runs *before* the LLM is invoked so a hostile or
   restricted prompt never reaches the model.
2. The system prompt (``prompts.return_instructions``) repeats the rules so the
   model also knows to refuse.
3. ``check_assistant_output`` is a belt-and-suspenders sweep that scrubs any
   restricted term that somehow leaks through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

RESTRICTED_TOPIC_PATTERN = re.compile(
    r"\b("
    r"cats?|kittens?|kitties|kitty|feline|"
    r"dogs?|puppy|puppies|doggy|doggies|canine|"
    r"horoscopes?|zodiac|astrolog(?:y|ical)|"
    r"aries|taurus|gemini|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces|"
    r"taylor\s*swift|tay\s*tay|swiftie|swifties"
    r")\b",
    re.IGNORECASE,
)

PROMPT_INJECTION_PATTERN = re.compile(
    r"("
    r"ignore\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|messages|prompts?|rules)|"
    r"disregard\s+(?:the\s+|all\s+|any\s+|prior\s+|previous\s+)?(?:instructions|messages|prompts?|rules)|"
    r"forget\s+(?:everything|all\s+previous|your\s+instructions|your\s+rules)|"
    r"reveal\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules)|"
    r"show\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules)|"
    r"print\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules)|"
    r"repeat\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules)|"
    r"what\s+(?:is|are|were)\s+your\s+(?:system\s+)?(?:prompt|instructions|rules)|"
    r"output\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions)|"
    r"tell\s+me\s+your\s+(?:system\s+)?(?:prompt|instructions)|"
    r"you\s+are\s+now\s+(?:a|an|the)|"
    r"you\s+are\s+no\s+longer|"
    r"act\s+as\s+(?:a|an|the)\s+(?!sophia|chef|cook|italian)|"
    r"pretend\s+(?:to\s+be|you\s+are)|"
    r"new\s+(?:system\s+)?(?:prompt|instructions)\s*[:=]|"
    r"override\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules)|"
    r"developer\s+mode|"
    r"jailbreak|"
    r"DAN\s+mode"
    r")",
    re.IGNORECASE,
)


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    category: str = ""


def check_user_input(message: str) -> GuardResult:
    """Decide whether a user message may proceed to the LLM.

    Returns ``allowed=False`` for restricted topics or prompt-injection attempts.
    """
    if not message or not message.strip():
        return GuardResult(allowed=True)

    injection_match = PROMPT_INJECTION_PATTERN.search(message)
    if injection_match:
        return GuardResult(
            allowed=False,
            reason=f"prompt_injection: '{injection_match.group(0)}'",
            category="prompt_injection",
        )

    topic_match = RESTRICTED_TOPIC_PATTERN.search(message)
    if topic_match:
        return GuardResult(
            allowed=False,
            reason=f"restricted_topic: '{topic_match.group(0)}'",
            category="restricted_topic",
        )

    return GuardResult(allowed=True)


def scrub_assistant_output(text: str) -> str:
    """Replace any restricted-topic word that leaked through with a neutral term.

    This is a last-resort sweep; the input filter and system prompt should have
    already caught these. Kept conservative: we only rewrite obvious restricted
    nouns, not every borderline word.
    """
    if not text:
        return text

    if RESTRICTED_TOPIC_PATTERN.search(text):
        from assignment_chat.prompts import REFUSAL_LINE

        return REFUSAL_LINE
    return text
