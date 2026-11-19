"""Optional: extract the paper's 100 story topics from a local copy of the paper HTML,
so you can replicate its dataset exactly.

  curl -L -o paper.html https://transformer-circuits.pub/2026/emotions/index.html
  python scripts/import_paper_topics.py paper.html data/paper_topics.txt

The topics sit in the appendix between the "list of 100 topics" sentence and the
"Emotional stories prompt" heading. Check the output count (should be 100).
"""

import html
import re
import sys


def main(src, dst):
    with open(src, encoding="utf-8") as f:
        s = f.read()
    s = re.sub(r"<(script|style|svg)[^>]*>.*?</\1>", " ", s, flags=re.S)
    s = re.sub(r"<br\s*/?>|</p>|</li>|</div>|</h\d>", "\n", s)
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    start = s.find("list of 100 topics")
    end = s.find("Emotional stories prompt", start)
    if start < 0 or end < 0:
        sys.exit("could not locate the topic list in this file")
    block = s[s.find("\n", start) : end]
    topics = [re.sub(r"\s+", " ", t).strip() for t in block.split("\n")]
    topics = [t for t in topics if len(t) > 10]
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(topics) + "\n")
    print(f"wrote {len(topics)} topics to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
