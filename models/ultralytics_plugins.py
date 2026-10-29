"""Runtime registration for custom Ultralytics YAML modules.

Ultralytics resolves YAML module names from ``ultralytics.nn.tasks``.  This
small adapter registers DMARE without modifying the installed Python package.
"""

from __future__ import annotations

import inspect
import textwrap

from .dmare import DMARE


def register_dmare() -> None:
    """Make ``DMARE`` available to Ultralytics' model YAML parser."""
    from ultralytics.nn import tasks

    if getattr(tasks, "_vision_detect_dmare_registered", False):
        return
    tasks.DMARE = DMARE
    source = textwrap.dedent(inspect.getsource(tasks.parse_model))
    if "            DMARE," not in source:
        marker = "            A2C2f,\n"
        if marker not in source:
            raise RuntimeError("Unsupported Ultralytics version: cannot register DMARE")
        source = source.replace(marker, marker + "            DMARE,\n", 1)
    namespace = tasks.__dict__
    exec(compile(source, "<vision_detect_parse_model>", "exec"), namespace)
    tasks._vision_detect_dmare_registered = True
