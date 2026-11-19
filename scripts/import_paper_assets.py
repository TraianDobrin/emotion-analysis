"""Extract the paper's 100 story topics and its emotional-story prompt from a local copy
of the paper HTML, for an exact replication of its dataset recipe.

  curl -L -o paper.html https://transformer-circuits.pub/2026/emotions/index.html
  python scripts/import_paper_assets.py paper.html data/

Writes data/paper_topics.txt (should be 100 lines) and data/paper_story_prompt.txt
(a template with {n_stories}, {topic}, {emotion}). Use them with
  generate_data.py --topics-file data/paper_topics.txt --story-prompt-file data/paper_story_prompt.txt
"""

import html
import os
import re
import sys


def page_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        s = f.read()
    s = re.sub(r"<(script|style|svg)[^>]*>.*?</\1>", " ", s, flags=re.S)
    s = re.sub(r"<br\s*/?>|</p>|</li>|</div>|</h\d>", "\n", s)
    return html.unescape(re.sub(r"<[^>]+>", " ", s))


def between(s: str, start_marker: str, end_marker: str) -> list[str]:
    i = s.find(start_marker)
    j = s.find(end_marker, i)
    if i < 0 or j < 0:
        sys.exit(f"could not find section starting with {start_marker!r}")
    block = s[s.find("\n", i): j]
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in block.split("\n")]
    return [line for line in lines if line]


def main(src: str, out_dir: str):
    s = page_text(src)
    os.makedirs(out_dir, exist_ok=True)

    topics = [t for t in between(s, "list of 100 topics", "Emotional stories prompt") if len(t) > 10]
    with open(os.path.join(out_dir, "paper_topics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(topics) + "\n")
    print(f"wrote {len(topics)} topics")

    prompt_lines = between(s, "Emotional stories prompt", "Neutral dialogues prompt")
    prompt = "\n".join(prompt_lines)
    for field in ("{n_stories}", "{topic}", "{emotion}"):
        if field not in prompt:
            sys.exit(f"story prompt is missing {field}; the page layout may have changed")
    with open(os.path.join(out_dir, "paper_story_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt + "\n")
    print(f"wrote story prompt ({len(prompt.split())} words)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
