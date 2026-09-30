"""
Browser model folders for deepfake-detection-38502 (see webfolder.py for the contract).

    decision_onnx.py    Julia 1 (Jev-like) with the question baked into the official ONNX graph
    classifier_onnx.py  off-the-shelf Hugging Face classifiers        (needs transformers<5)
    saved_onnx.py       the project's saved embedding detectors        (needs transformers<5)
    quantize.py         weight-only int8 with standard ONNX ops (WebAssembly-safe, activations stay fp32)
    torch_onnx.py       tracing helpers and the parity check
"""
