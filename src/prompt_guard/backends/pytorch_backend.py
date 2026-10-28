"""PyTorch reference and GPU backend.

Imported only when the pytorch runtime is selected. The CPU image does not
install torch, so this module is optional.
"""

from collections.abc import Callable

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

from prompt_guard.errors import FatalConfigError, scan_timeout
from prompt_guard.scanner import LogitsModel

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_DTYPE_NAMES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


class PytorchBackend(LogitsModel):
    def __init__(self, snapshot: str, device: str, dtype_name: str) -> None:
        if dtype_name not in _DTYPES:
            raise FatalConfigError(f"Unsupported dtype {dtype_name}.")
        if device == "cuda" and not torch.cuda.is_available():
            raise FatalConfigError("CUDA was requested but no GPU is available to PyTorch.")
        if device == "cpu" and dtype_name != "float32":
            raise FatalConfigError("CPU PyTorch inference uses float32.")
        requested = _DTYPES[dtype_name]
        model = AutoModelForSequenceClassification.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            dtype=requested,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to(device)
        torch.set_grad_enabled(False)
        self.model = model
        self.device = device
        self.torch_version = torch.__version__
        self.gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
        self.quantization = _actual_dtype(model)
        self.providers = (device,)

    def predict_logits(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        should_stop: Callable[[], bool],
    ) -> np.ndarray:
        if should_stop():
            raise scan_timeout()
        ids = torch.from_numpy(np.ascontiguousarray(input_ids, dtype=np.int64)).to(self.device)
        mask = torch.from_numpy(np.ascontiguousarray(attention_mask, dtype=np.int64)).to(self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=ids, attention_mask=mask).logits
        if should_stop():
            raise scan_timeout()
        return logits.float().detach().cpu().numpy()


def _actual_dtype(model: torch.nn.Module) -> str:
    names = {_DTYPE_NAMES.get(parameter.dtype, str(parameter.dtype)) for parameter in model.parameters()}
    if len(names) == 1:
        return names.pop()
    return "mixed:" + ",".join(sorted(names))
