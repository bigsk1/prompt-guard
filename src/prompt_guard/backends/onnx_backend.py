"""ONNX Runtime backend. Session creation does not execute model Python."""

import threading
from collections.abc import Callable

import numpy as np
import onnxruntime as ort

from prompt_guard.errors import FatalConfigError, scan_timeout
from prompt_guard.scanner import LogitsModel

_NUMPY_TYPES = {
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


class OnnxBackend(LogitsModel):
    def __init__(self, model_path: str, intra_op_threads: int) -> None:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        if intra_op_threads > 0:
            options.intra_op_num_threads = intra_op_threads
        self.intra_op_threads = intra_op_threads if intra_op_threads > 0 else None
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        providers = list(self.session.get_providers())
        if providers != ["CPUExecutionProvider"]:
            raise FatalConfigError(f"ONNX session providers were {providers}, expected CPU only.")
        inputs = {item.name: item.type for item in self.session.get_inputs()}
        if set(inputs) != {"input_ids", "attention_mask"}:
            raise FatalConfigError(
                "ONNX model inputs must be exactly input_ids and attention_mask."
            )
        self._id_dtype = _NUMPY_TYPES.get(inputs["input_ids"])
        self._mask_dtype = _NUMPY_TYPES.get(inputs["attention_mask"])
        if self._id_dtype is None or self._mask_dtype is None:
            raise FatalConfigError("ONNX input types must be int32 or int64.")
        outputs = [item.name for item in self.session.get_outputs()]
        if not outputs or outputs[0] != "logits":
            raise FatalConfigError("ONNX model output must be named logits.")
        self.providers = tuple(providers)
        self.version = ort.__version__

    def predict_logits(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        should_stop: Callable[[], bool],
    ) -> np.ndarray:
        if should_stop():
            raise scan_timeout()
        options = ort.RunOptions()
        done = threading.Event()

        def watch() -> None:
            while not done.wait(0.05):
                if should_stop():
                    options.terminate = True
                    return

        watcher = threading.Thread(target=watch, name="prompt-guard-deadline", daemon=True)
        watcher.start()
        try:
            if should_stop():
                options.terminate = True
                raise scan_timeout()
            result = self.session.run(
                None,
                {
                    "input_ids": np.asarray(input_ids, dtype=self._id_dtype),
                    "attention_mask": np.asarray(attention_mask, dtype=self._mask_dtype),
                },
                options,
            )
        except scan_timeout:
            raise
        except Exception as exc:
            if should_stop() or getattr(options, "terminate", False):
                raise scan_timeout() from None
            raise RuntimeError("ONNX inference failed.") from exc
        finally:
            done.set()
            watcher.join(timeout=1)
        if should_stop():
            raise scan_timeout()
        return np.asarray(result[0])
