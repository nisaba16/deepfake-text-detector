"""
Jev-like zero-shot baseline. Moved to textdet.detectors.decision, which adds the Julia 1 backend next to
Laya; these names are kept so existing imports keep working.
"""
from textdet.detectors.decision import (DEFAULT_CHECKPOINT, QUESTION_PRESETS, JuliaDetector,  # noqa: F401
                                        LayaDetector, _capture_option_logits)

JevLikeDetector = LayaDetector
