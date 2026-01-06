import abc
import numpy as np
from typing import Any, List

class EncryptionScheme(abc.ABC):
    """
    Abstract base class for encryption schemes in federated learning.
    """

    @abc.abstractmethod
    def encrypt(self, params: np.ndarray) -> Any:
        """
        Encrypts the model parameters.
        """
        pass

    @abc.abstractmethod
    def decrypt(self, encrypted_params: Any) -> np.ndarray:
        """
        Decrypts the model parameters.
        """
        pass

    @abc.abstractmethod
    def aggregate(self, encrypted_params_list: List[Any]) -> Any:
        """
        Aggregates a list of encrypted parameters. 
        Note: In homomorphic encryption, this would be performed on ciphertexts.
        """
        pass

class NoEncryption(EncryptionScheme):
    """
    A dummy encryption scheme that does nothing.
    Useful for baseline comparisons and debugging.
    """

    def encrypt(self, params: np.ndarray) -> np.ndarray:
        return params

    def decrypt(self, encrypted_params: np.ndarray) -> np.ndarray:
        return encrypted_params

    def aggregate(self, encrypted_params_list: List[np.ndarray]) -> np.ndarray:
        """
        For no encryption, aggregation is typically handled by the Orchestrator's strategy via decryption,
        but if we treat 'aggregate' as a homomorphic operation, this would be simple summation or similar.
        However, usually extraction of the mean happens at the Orchestrator level.
        
        To be consistent with 'encrypted' aggregation:
        If this was FHE, we would sum the ciphertexts.
        Here we just return the list or sum them?
        
        Let's assume the Orchestrator uses an AggregationStrategy that might call this,
        OR the Orchestrator delegates the mathematical operation to the scheme if it's FHE.
        
        For simplicity in this interface: let's perform a summation here, as that is the standard FHE operation.
        The Orchestrator will likely want to average, which involves division. 
        Division is often hard in FHE. Usually we sum and then the decryption side (if it has the key) divides,
        or we multiply by 1/N.
        
        Let's implement simple element-wise summation here.
        """
        if not encrypted_params_list:
            return None
        return sum(encrypted_params_list)
