"""transformers>=5 compatibility shim for assistant-axis.

assistant-axis targets transformers 4.x, where
``apply_chat_template(..., tokenize=True)`` returns ``list[int]``. In 5.x it
returns a ``BatchEncoding`` by default, which breaks its ``ConversationEncoder``
and activation extractor. jlens needs transformers>=5.5, so restore the old
default instead of pinning.

Run an assistant-axis pipeline script with the patch applied::

    python compat.py path/to/script.py [script args...]
"""

from __future__ import annotations

import functools
import runpy
import sys

from transformers import PreTrainedTokenizerBase


def patch_apply_chat_template() -> None:
    orig = PreTrainedTokenizerBase.apply_chat_template
    if getattr(orig, "_persona_drift_patched", False):
        return

    @functools.wraps(orig)
    def apply_chat_template(self, conversation, *args, **kwargs):
        if kwargs.get("tokenize", True) and "return_dict" not in kwargs:
            kwargs["return_dict"] = False
        return orig(self, conversation, *args, **kwargs)

    apply_chat_template._persona_drift_patched = True
    PreTrainedTokenizerBase.apply_chat_template = apply_chat_template


if __name__ == "__main__":
    patch_apply_chat_template()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
