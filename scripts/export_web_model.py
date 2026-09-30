#!/usr/bin/env python3
"""
Export a detector as a model folder of the browser app (deepfake-detection-38502/public/models/<id>/),
then check that the exported folder, run exactly as the page runs it, agrees with the Python detector.

    julia                    Julia 1 with one question baked in (weights from the official ONNX export;
                             transformers version does not matter)
    hf:<repo or short name>  an off-the-shelf classifier, e.g. hf:e5-small-raid, hf:tmr-roberta-raid
    saved:<detector.pkl>     a detector trained here (embedding + LR / one-class SVM)

hf: and saved: use torch.onnx.export, which needs transformers<5 (requirements-export.txt).

Examples:
    python scripts/export_web_model.py julia --id julia1-jev
    .venv-export/bin/python scripts/export_web_model.py hf:e5-small-raid --id e5-small-raid
    .venv-export/bin/python scripts/export_web_model.py \\
        saved:saved_models/human_ai_sentence-transformers_all-MiniLM-L6-v2_embedding_layer2_last_ocsvm.pkl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textdet.data import prepare, resolve
from textdet.detectors.base import safe_name
from textdet.detectors.decision import QUESTION_PRESETS
from textdet.export.webfolder import FRONTEND_MODELS


def main():
    parser = argparse.ArgumentParser(description="Export a detector for client-side inference in the browser")
    parser.add_argument("detector", help="julia | hf:<repo or short name> | saved:<detector.pkl>")
    parser.add_argument("--id", type=str, default=None, help="Folder name under the app's public/models/")
    parser.add_argument("--models_dir", type=str, default=str(FRONTEND_MODELS),
                        help="The app's public/models directory (index.json is updated there)")
    parser.add_argument("--precision", type=str, default=None,
                        help="julia: mixed (default) | int8 | fp32; hf: int8 (default) | fp16 | fp32; saved: int8 | fp32")
    parser.add_argument("--question", type=str, default="ai_generated", choices=sorted(QUESTION_PRESETS),
                        help="julia: the question baked into the graph")
    parser.add_argument("--max_length", type=int, default=None, help="Tokens read (julia: sequence incl. question)")
    parser.add_argument("--name", type=str, default=None, help="Name shown in the model picker")
    parser.add_argument("--description", type=str, default=None, help="Text shown under the model picker")
    parser.add_argument("--no_register", action="store_true", help="Do not add the folder to index.json")
    parser.add_argument("--calibrate_fpr", type=float, default=None,
                        help="Shift the output so the page's 50%% line flags this fraction of human texts "
                             "(e.g. 0.05). Detectors are overconfident on human text of new domains.")
    parser.add_argument("--calibration_dataset", type=str, default="mage_test",
                        help="Dataset whose HUMAN texts set the calibration (use your own domain when you can)")
    parser.add_argument("--calibration_n", type=int, default=2000, help="Human texts used for the calibration")
    parser.add_argument("--parity_dataset", type=str, default="mage_ood_gpt4",
                        help="Dataset whose texts are used for the parity check ('' to skip)")
    parser.add_argument("--parity_n", type=int, default=40, help="Texts in the parity check")
    args = parser.parse_args()

    kind, _, arg = args.detector.partition(":")
    folder_id = args.id or safe_name(args.detector.split("/")[-1].replace(".pkl", "")).lower()
    out = Path(args.models_dir) / folder_id
    common = dict(name=args.name, description=args.description, register=not args.no_register)

    if kind == "julia":
        from textdet.export.decision_onnx import export_decision_web
        kw = dict(question=QUESTION_PRESETS[args.question], precision=args.precision or "mixed", **common)
        if arg:
            kw["checkpoint"] = arg
        if args.max_length:
            kw["max_length"] = args.max_length
        export_decision_web(out, **kw)
    elif kind == "hf":
        from textdet.export.classifier_onnx import export_classifier_web
        export_classifier_web(arg, out, precision=args.precision or "int8",
                              **({"max_length": args.max_length} if args.max_length else {}), **common)
    elif kind == "saved":
        from textdet.export.saved_onnx import export_saved_web
        export_saved_web(arg, out, precision=args.precision or "int8", max_length=args.max_length, **common)
    else:
        raise SystemExit(f"Unknown detector {args.detector!r}: julia, hf:<repo> or saved:<pkl>")

    if args.parity_dataset:
        from textdet.detectors.registry import build_detector
        from textdet.detectors.webmodel import WebModelDetector
        from textdet.export.torch_onnx import parity
        texts = prepare(resolve(args.parity_dataset), n_rows=args.parity_n, stratified=True, seed=0).texts
        reference = build_detector(args.detector, device="cpu",
                                   questions={args.question: QUESTION_PRESETS[args.question]} if kind == "julia" else None)
        result = parity(reference, WebModelDetector(str(out)), texts)
        (out / "parity.json").write_text(json.dumps({"reference": args.detector, "dataset": args.parity_dataset,
                                                     **result}, indent=2) + "\n")

    if args.calibrate_fpr:
        # After the parity check: calibration deliberately shifts the scores
        from textdet.export.calibrate import calibrate_web_model
        ts = resolve(args.calibration_dataset)
        human = ts.subset([i for i in range(len(ts)) if ts.labels[i] == 0])
        human = prepare(human, n_rows=args.calibration_n, seed=0)
        calibrate_web_model(out, human.texts, args.calibrate_fpr, source=args.calibration_dataset)


if __name__ == "__main__":
    main()
