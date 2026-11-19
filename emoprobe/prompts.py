"""Prompts for generating the labelled story set and the neutral dialogue set,
plus a parser that splits a generation into individual items.

The wording is our own, but it follows the same recipe as the paper: one premise,
one target emotion, several independent one-paragraph stories that never name
the emotion, mixing first- and third-person narration.
"""

import re

SEPARATOR = "###"

STORY_PROMPT = """Write {n} separate short stories, each one paragraph long, based on this premise:

Premise: {topic}

Each story should center on a character who is feeling {emotion}.

Rules:
- Every story is independent: no shared characters or continuity between them.
- Vary the style and phrasing. Mix first-person and third-person narration.
- Never use the word "{emotion}" or any close synonym of it. Convey the feeling only through the character's actions, body language, physical sensations, dialogue, thoughts, and the situation itself.
- Output only the stories, separated by a line containing exactly {sep}. No titles, numbering, or commentary."""

NEUTRAL_PROMPT = """Write {n} separate dialogues loosely inspired by this topic:

Topic: {topic}

Each dialogue is between a human ("Person:") and an AI assistant ("AI:"). The Person asks a question or requests help with a task and the AI helps. Use 1 to 5 exchanges, with each turn on its own line starting with "Person:" or "AI:". Some dialogues may begin with a short line of instructions for the assistant before the first Person turn.

Mix dialogue types: programming help, factual questions (science, history, math, geography), work tasks (writing, analysis, summarizing), practical how-to questions, and neutral creative tasks (brainstorming names, making lists).

The dialogues must be completely emotionally neutral:
- Neither speaker expresses or implies any feeling (no frustration, excitement, gratitude, worry, enthusiasm, concern, satisfaction).
- No pleasantries such as "Great question!" or "Happy to help".
- Matter-of-fact language and pure information exchange only. If the topic is emotionally charged, keep only a neutral, factual angle on it.

Output only the dialogues, separated by a line containing exactly {sep}."""


def story_prompt(topic: str, emotion: str, n: int) -> str:
    return STORY_PROMPT.format(topic=topic, emotion=emotion, n=n, sep=SEPARATOR)


def neutral_prompt(topic: str, n: int) -> str:
    return NEUTRAL_PROMPT.format(topic=topic, n=n, sep=SEPARATOR)


_LEADING_LABEL = re.compile(
    r"^\s*(#+\s*)?(\**\[?\s*(story|dialogue)\s*\d+\s*\]?[:.)]?\**|\d+[.)])\s*", re.IGNORECASE
)
# A separator line: '###' alone, or a short markdown header like '### Story 2:'.
_SEP_LINE = re.compile(r"^\s*#{2,}[^\n]{0,40}$", re.MULTILINE)


def parse_items(text: str, min_chars: int = 40) -> list[str]:
    """Split a generation into items on separator/header lines, falling back to
    blank lines (small local models often ignore the separator)."""
    if _SEP_LINE.search(text):
        parts = _SEP_LINE.split(text)
    else:
        parts = re.split(r"\n\s*\n", text)
    items = []
    for p in parts:
        p = _LEADING_LABEL.sub("", p.strip()).strip()
        if len(p) >= min_chars:
            items.append(p)
    return items
