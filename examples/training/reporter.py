"""Small atomic progress-file writer for a training process (Python stdlib only).

Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
"""

import json
import os
from pathlib import Path
import tempfile


def process_start_ticks(pid=None):
    """Return Linux's PID reuse guard, or None where /proc is unavailable."""
    try:
        fields = Path(f"/proc/{pid or os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


class ProgressReporter:
    """One writer per output directory; callers supply the measured stage data.

    The process calling this constructor owns the reported work. Construct it
    inside the worker process, rather than in a parent before spawning workers.
    File readers need the same user/process namespace to establish liveness.
    """

    def __init__(self, output_dir, run_id, campaign, source_id="trainer-0",
                 source_label="Training worker", model=None):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.document = {
            "schema_version": 1,
            "run_id": run_id,
            "campaign": campaign,
            "source_id": source_id,
            "source_label": source_label,
            "pid": os.getpid(),
        }
        ticks = process_start_ticks()
        if ticks is not None:
            self.document["process_start_ticks"] = ticks
        if model is not None:
            self.document["model"] = model

    def write(self, *, data=(), training=(), evaluation=()):
        """Replace one complete snapshot; no counters, scores or times are guessed.

        Report state='complete' after successful completion and state='failed'
        on a caught failure. A process exit also prevents stale running rows
        from establishing activity, even if no final write was possible.
        """
        document = {
            **self.document,
            "stages": {
                "data": list(data),
                "training": list(training),
                "evaluation": list(evaluation),
            },
        }
        payload = json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             prefix=".training-progress-", suffix=".tmp",
                                             dir=self.output_dir, delete=False) as output:
                temporary = Path(output.name)
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.output_dir / "training-progress.json")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
