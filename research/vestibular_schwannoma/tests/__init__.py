"""Tests for the vestibular schwannoma research project."""

import sys
from pathlib import Path

# Tests import both `vestibular_schwannoma.*` and the project-local `workflow.*`
# and `scripts.*`, so put research/ and the project directory on sys.path.
_PROJECT = Path(__file__).resolve().parents[1]
for _root in (_PROJECT.parent, _PROJECT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
