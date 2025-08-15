from typing import Protocol, Any
import numpy as np

class ThetaEmbedder(Protocol):
    def transform(self, X: np.ndarray) -> np.ndarray: ...

class AnsatzFactory(Protocol):
    def qaoa(self, n_qubits: int, p: int) -> Any: ...
    def twolocal(self, n_qubits: int, reps: int) -> Any: ...

class Estimator(Protocol):
    def run(self, circuit: Any, observable: Any, params: np.ndarray) -> float: ...

def try_import_core():
    try:
        import compiled_core.quantum_core as qc
        return qc
    except Exception:
        return None
