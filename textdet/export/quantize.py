"""
Weight-only int8 quantization with standard ONNX ops (runs in onnxruntime-web's WebAssembly backend).

Why not onnxruntime's quantize_dynamic: it also quantizes the activations of every MatMul at run time
(per tensor, uint8). BERT-family encoders have outlier activation dimensions, and on Julia 1 (mmBERT) that
destroys the output (rank correlation 0.02 with fp32 on MAGE texts). Here only the weights are int8, with one
scale per block of `block` weights along the reduction axis (one per column / per row when block is None):

    MatMul weight [K, N]     int8 [K/b, b, N] * scale [K/b, 1, N] -> Reshape [K, N] -> MatMul
                             (constant subgraph: onnxruntime folds it once when the session is created)
    embedding table [V, D]   Gather int8 rows [.., D/b, b] and their scales [.., D/b, 1], Cast, Mul, Reshape:
                             the table is never expanded to fp32

Symmetric (zero point 0). Download size ~1/4 of fp32 (+ 1/16 for block 64 scales); compute stays fp32.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _blocked_int8(w: np.ndarray, axis: int, block: Optional[int]):
    """Symmetric int8 of a 2-D array with one scale per `block` values along `axis` (all of it when None).

    Returns (q, scale) with q reshaped to expose the blocks: axis 0 -> q [K/b, b, N], scale [K/b, 1, N];
    axis 1 -> q [V, D/b, b], scale [V, D/b, 1].
    """
    k = w.shape[axis]
    b = block if block and k % block == 0 else k
    if axis == 0:
        blocks = w.reshape(k // b, b, w.shape[1])
        scale = np.abs(blocks).max(axis=1, keepdims=True) / 127.0
    else:
        blocks = w.reshape(w.shape[0], k // b, b)
        scale = np.abs(blocks).max(axis=2, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.clip(np.rint(blocks / scale), -127, 127).astype(np.int8)
    return q, scale


def quantize_weights_int8(src: Path, dst: Path, block: Optional[int] = 32, min_elements: int = 4096,
                          gather_min_rows: int = 1024, matmul: str = "int8", gather: bool = True) -> Dict[str, int]:
    """Quantize the MatMul weights and the large embedding Gathers of `src` into a single file `dst`.

    block: weights per scale along the reduction axis (None: one scale per column / row).
    min_elements: smaller MatMul weights stay fp32.  gather_min_rows: Gather tables with fewer rows
    (type embeddings, small lookups) stay fp32.
    matmul: "int8", "fp16" (stored half precision, Cast to fp32) or "fp32" (unchanged).
    gather: quantize large embedding tables.  Returns counts of what was converted.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(src), load_external_data=True)
    g = model.graph
    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    if opset < 13:
        raise ValueError("Needs opset >= 13")
    inits = {t.name: t for t in g.initializer}
    consumers: Dict[str, list] = {}
    for node in g.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    counts = {"matmul": 0, "gather": 0}
    new_inits, head_nodes, drop = [], [], set()
    tails: Dict[str, list] = {}   # new output of a quantized Gather -> nodes to insert right after it

    def const(array, name):
        new_inits.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def only_used_as(name: str, op: str, position: int) -> bool:
        users = consumers.get(name, [])
        return bool(users) and all(n.op_type == op and list(n.input).index(name) == position for n in users)

    for name, tensor in inits.items():
        if tensor.data_type != TensorProto.FLOAT:
            continue
        dims = list(tensor.dims)
        if (matmul != "fp32" and len(dims) == 2 and dims[0] * dims[1] >= min_elements
                and only_used_as(name, "MatMul", 1)):
            w = numpy_helper.to_array(tensor)
            if matmul == "fp16":
                const(w.astype(np.float16), name + "_half")
                head_nodes.append(helper.make_node("Cast", [name + "_half"], [name], to=TensorProto.FLOAT))
            else:
                q, scale = _blocked_int8(w, axis=0, block=block)
                head_nodes += [
                    helper.make_node("Cast", [const(q, name + "_q")], [name + "_qf"], to=TensorProto.FLOAT),
                    helper.make_node("Mul", [name + "_qf", const(scale, name + "_scale")], [name + "_blocks"]),
                    helper.make_node("Reshape", [name + "_blocks", const(np.array(dims, dtype=np.int64),
                                                                         name + "_shape")], [name]),
                ]
            drop.add(name)
            counts["matmul"] += 1
        elif gather and len(dims) == 2 and dims[0] >= gather_min_rows and only_used_as(name, "Gather", 0):
            if any(next((a.i for a in n.attribute if a.name == "axis"), 0) != 0 for n in consumers[name]):
                continue
            q, scale = _blocked_int8(numpy_helper.to_array(tensor), axis=1, block=block)
            const(q, name + "_q")
            const(scale, name + "_scale")
            width = const(np.array([dims[1]], dtype=np.int64), name + "_width")
            for n in consumers[name]:
                ids, out = n.input[1], n.output[0]
                tag = f"{out}_q"
                n.input[0], n.output[0] = name + "_q", tag + "_int8"
                tails[tag + "_int8"] = [
                    helper.make_node("Gather", [name + "_scale", ids], [tag + "_scale"], axis=0),
                    helper.make_node("Cast", [tag + "_int8"], [tag + "_float"], to=TensorProto.FLOAT),
                    helper.make_node("Mul", [tag + "_float", tag + "_scale"], [tag + "_blocks"]),
                    helper.make_node("Shape", [ids], [tag + "_ids_shape"]),
                    helper.make_node("Concat", [tag + "_ids_shape", width], [tag + "_shape"], axis=0),
                    helper.make_node("Reshape", [tag + "_blocks", tag + "_shape"], [out]),
                ]
            drop.add(name)
            counts["gather"] += 1

    kept = [t for t in g.initializer if t.name not in drop]
    del g.initializer[:]
    g.initializer.extend(kept + new_inits)
    # Topological order: weight subgraphs first (they only read initializers), Gather tails after their Gather
    order = list(head_nodes)
    for node in g.node:
        order.append(node)
        order += tails.get(node.output[0] if node.output else "", [])
    del g.node[:]
    g.node.extend(order)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst), save_as_external_data=False)
    return counts
