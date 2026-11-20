"""The neutral-dialogue prompt, and a parser that splits one LLM response into items.

The story prompt is the paper's own, imported by scripts/import_paper_assets.py.
"""

import re

NEUTRAL_PROMPT = """Write {n} separate dialogues loosely inspired by this topic:

Topic: {topic}

Each dialogue is between a human ("Person:") and an AI assistant ("AI:"). The Person asks a question or requests help with a task and the AI helps. Use 1 to 5 exchanges, with each turn on its own line starting with "Person:" or "AI:".

Mix dialogue types: programming help, factual questions (science, history, math, geography), work tasks (writing, analysis, summarizing), practical how-to questions, and neutral creative tasks (brainstorming names, making lists).

The dialogues must be completely emotionally neutral:
- Neither speaker expresses or implies any feeling (no frustration, excitement, gratitude, worry, enthusiasm, concern, satisfaction).
- No pleasantries such as "Great question!" or "Happy to help".
- Matter-of-fact language and pure information exchange only. If the topic is emotionally charged, keep only a neutral, factual angle on it.

Output only the dialogues, separated by a line containing exactly ###."""


def neutral_prompt(topic, n):
    return NEUTRAL_PROMPT.format(topic=topic, n=n)


# A line that separates items: '[story 2]', 'Story 2:', '**Dialogue 3**', '###', '### Story 2'.
_SEPARATOR = re.compile(r"^\s*(?:\**\[?\s*(?:story|dialogue)\s*\d+\s*\]?[:.]?\**|#{2,}.{0,40})\s*$",
                        re.IGNORECASE | re.MULTILINE)


def parse_items(text, min_chars=40):
    """Split a response on separator lines, or on blank lines if there are none.
    Fragments shorter than min_chars (stray headers, 'Here are your stories:') are dropped."""
    parts = _SEPARATOR.split(text) if _SEPARATOR.search(text) else re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if len(p.strip()) >= min_chars]
