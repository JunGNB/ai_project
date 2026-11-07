"""Offline token-count worker for the MiniCPM tokenizer shipped with oMLX.

The benchmark records token counts as measured by oMLX's ``/v1/tokenize``
endpoint.  For reproducible offline rebuilds this worker reproduces exactly the
same numbers with the model's own ``tokenizer.json`` and ``chat_template.jinja``
(both are part of the oMLX model download):

* ``messages`` payloads: render the chat template with
  ``add_generation_prompt=True`` and ``enable_thinking=True`` (the defaults the
  oMLX endpoint applies), then encode with special tokens;
* ``text`` payloads: encode the text with special tokens.

The counts were validated against the existing oMLX tokenize cache: all 2,600
payloads reconstructed from the previous test sets match exactly.

The script is executed by the interpreter bundled with oMLX.app (Python 3.11 +
``tokenizers`` + ``jinja2``), not by the project venv.  Protocol: one JSON object
per line on stdin, one JSON object per line on stdout::

    {"kind": "messages", "messages": [...]}       -> {"count": N}
    {"kind": "text", "text": "..."}               -> {"count": N}
"""

from __future__ import annotations

import json
import os
import sys

MODEL_DIR = os.environ.get("MB_TOKENIZER_DIR", os.path.expanduser("~/.omlx/models/openbmb/MiniCPM5-1B"))


def _load():
    from tokenizers import Tokenizer
    import jinja2

    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    with open(os.path.join(MODEL_DIR, "chat_template.jinja"), encoding="utf-8") as fh:
        template = jinja2.Environment().from_string(fh.read())
    return tokenizer, template


def main() -> int:
    tokenizer, template = _load()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if item.get("kind") == "messages":
                rendered = template.render(
                    messages=item["messages"],
                    tools=None,
                    bos_token="<s>",
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                count = len(tokenizer.encode(rendered, add_special_tokens=True).ids)
            elif item.get("kind") == "text":
                count = len(tokenizer.encode(item["text"], add_special_tokens=True).ids)
            else:
                raise ValueError(f"unknown kind: {item.get('kind')!r}")
            sys.stdout.write(json.dumps({"count": count}) + "\n")
        except Exception as exc:  # noqa: BLE001 - report, do not die mid-stream
            sys.stdout.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
