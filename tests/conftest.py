import os
import sys
import types
from pathlib import Path


# Ensure src/ is on sys.path for test imports without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Default to light tests to avoid heavy deps in CI or constrained envs
os.environ.setdefault("LIGHT_TESTS", "1")


# Provide a minimal faiss stub if faiss-cpu is not installed
try:
    import faiss as _faiss  # type: ignore
except Exception:  # pragma: no cover
    import numpy as _np

    class IndexFlatIP:  # minimal stub
        def __init__(self, dim: int):
            self.dim = dim
            self._vecs = _np.zeros((0, dim), dtype=_np.float32)

        def add(self, arr):
            arr = _np.asarray(arr, dtype=_np.float32)
            if arr.shape[1] != self.dim:
                raise ValueError("dimension mismatch")
            if self._vecs.size == 0:
                self._vecs = arr
            else:
                self._vecs = _np.vstack([self._vecs, arr])

        def search(self, q, k: int):
            q = _np.asarray(q, dtype=_np.float32)
            if self._vecs.size == 0:
                dists = _np.zeros((q.shape[0], k), dtype=_np.float32)
                idxs = -_np.ones((q.shape[0], k), dtype=_np.int64)
                return dists, idxs
            # cosine via inner product since inputs are pre-normalized in store
            sims = q @ self._vecs.T
            idxs = _np.argsort(-sims, axis=1)[:, :k]
            dists = _np.take_along_axis(sims, idxs, axis=1)
            return dists.astype(_np.float32), idxs.astype(_np.int64)

    def write_index(index: IndexFlatIP, path: str):  # noqa: N802
        import numpy as _np
        _np.save(path, index._vecs)

    def read_index(path: str) -> IndexFlatIP:  # noqa: N802
        import numpy as _np
        arr = _np.load(path, allow_pickle=False)
        idx = IndexFlatIP(arr.shape[1])
        idx.add(arr)
        return idx

    sys.modules["faiss"] = types.SimpleNamespace(
        IndexFlatIP=IndexFlatIP, write_index=write_index, read_index=read_index
    )


# Provide an openai stub if not installed
try:
    import openai as _openai  # type: ignore
except Exception:  # pragma: no cover
    class _Msg:
        def __init__(self, content: str = "{}"):
            self.message = types.SimpleNamespace(content=content)

    class _Completions:
        def create(self, **kwargs):
            return types.SimpleNamespace(choices=[_Msg()])

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Embeddings:
        def create(self, **kwargs):
            return types.SimpleNamespace(data=[])

    class OpenAI:  # noqa: N801
        def __init__(self, *args, **kwargs):
            self.chat = _Chat()
            self.embeddings = _Embeddings()

    sys.modules["openai"] = types.SimpleNamespace(OpenAI=OpenAI)


# Provide torch and transformers stubs if not installed
try:
    import torch as _torch  # type: ignore
except Exception:  # pragma: no cover
    import contextlib as _contextlib

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    @_contextlib.contextmanager
    def no_grad():  # noqa: N802
        yield

    sys.modules["torch"] = types.SimpleNamespace(cuda=_Cuda(), no_grad=no_grad)

try:
    import transformers as _transformers  # type: ignore
except Exception:  # pragma: no cover
    class _Tok:
        def __call__(self, *a, **k):
            # minimal structure with attention_mask for shape compatibility if ever used
            return types.SimpleNamespace(
                to=lambda device: types.SimpleNamespace(attention_mask=None)
            )

    class AutoTokenizer:  # noqa: N801
        @staticmethod
        def from_pretrained(name):
            return _Tok()

    class _Model:
        config = types.SimpleNamespace(hidden_size=128)
        def to(self, device):
            return self
        def eval(self):
            return None
        def __call__(self, **kwargs):
            return types.SimpleNamespace(last_hidden_state=None)

    class AutoModel:  # noqa: N801
        @staticmethod
        def from_pretrained(name):
            return _Model()

    sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=AutoTokenizer, AutoModel=AutoModel)


# Provide tenacity stub if not installed
try:
    import tenacity as _tenacity  # type: ignore
except Exception:  # pragma: no cover
    def retry(*a, **k):  # noqa: ANN001, D401
        """No-op retry decorator for tests."""
        def _wrap(fn):
            return fn
        return _wrap

    def stop_after_attempt(*a, **k):  # noqa: ANN001
        return None

    def wait_exponential(*a, **k):  # noqa: ANN001
        return None

    sys.modules["tenacity"] = types.SimpleNamespace(
        retry=retry, stop_after_attempt=stop_after_attempt, wait_exponential=wait_exponential
    )
