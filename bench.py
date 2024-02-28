from time import perf_counter
from contextlib import contextmanager

import numpy as np
import onnx
from spox import argument, Tensor, build, Var
import spox.opset.ai.onnx.v18 as op
import spox.opset.ai.onnx.ml.v3 as ml
import onnxruntime as ort


@contextmanager
def time_it(msg: str):
    t0 = perf_counter()
    try:
        yield
    finally:
        t1 = perf_counter()
        print(f"{msg}: {t1-t0:.3}s")
        


def parallel_subgraphs(var: Var, n: int):
    """Add ``N`` parallel subgraphs. """
    out = var
    for _ in range(n):
        strs = ml.label_encoder(var, keys_int64s=[1, 2, 3], values_strings=["a", "b", "c"])
        ints = ml.label_encoder(strs, keys_strings=["a", "b", "c"], values_int64s=[1, 2, 3])
        out = op.add(out, ints)
    return out


def main():
    n_parallel_subgraphs = 1000
    n_stages = 1

    print(f"{onnx.__version__=}")
    print(f"{ort.__version__=}")
    print(f"{n_parallel_subgraphs=}")
    print(f"{n_stages=}")
    print()

    with time_it("Constructing"):
        a = argument(Tensor(np.int64, shape=("N",)))
        cond = argument(Tensor(np.bool_, shape=()))

        b = a
        for _ in range(n_stages):
            (b,) = op.if_(
                cond,
                then_branch=lambda: [parallel_subgraphs(b, n_parallel_subgraphs)],
                else_branch=lambda: [parallel_subgraphs(b, n_parallel_subgraphs)],
            )
                

    with time_it("Building"):
        proto = build({"a": a, "cond": cond}, {"b": b})

    onnx.save(proto, "model.onnx")

    with time_it("Session creation"):
        ort.InferenceSession("model.onnx")


if __name__ == "__main__":
    main()
