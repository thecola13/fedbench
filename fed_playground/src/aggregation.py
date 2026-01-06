import abc
from typing import List, Any
import numpy as np
from .encryption import EncryptionScheme

class AggregationStrategy(abc.ABC):
    """
    Abstract base class for aggregation strategies in the Orchestrator.
    """
    @abc.abstractmethod
    def aggregate(self, encrypted_models: List[Any], encryption_scheme: EncryptionScheme) -> Any:
        """
        Aggregates the encrypted models.
        Returns the aggregated encrypted model (or decrypted if the scheme allows/requires).
        """
        pass

class MeanAggregation(AggregationStrategy):
    """
    Aggregates models by calculating the mean.
    For Homomorphic Encryption: Sums the models and typically the division by N 
    happens after decryption or by multiplying by 1/N if supported.
    
    For this playground, we assume the 'encryption_scheme.aggregate' performs summation.
    Then we might need to handle the division.
    
    If the encryption scheme is NoEncryption, we simply take the mean.
    If it is FHE, we might return the Sum and let the client divide, 
    or check if the scheme supports scalar multiplication.
    
    To keep it generic: We will ask the encryption scheme to 'aggregate' (sum) 
    and then we try to divide by N.
    """
    def aggregate(self, encrypted_models: List[Any], encryption_scheme: EncryptionScheme) -> Any:
        # Step 1: Summation via the encryption scheme
        # Most FHE schemes support efficient addition.
        if not encrypted_models:
            return None
            
        summed_model = encryption_scheme.aggregate(encrypted_models)
        
        # Step 2: Division for Mean
        # If we can access the underlying data (NoEncryption), we divide.
        # If it's real encryption, we might not be able to divide here without a key.
        # A common trick in Federated Learning with Secure Aggregation is to weight updates beforehand 
        # or have the clients divide before sending if specific requirements are met.
        # OR: The Orchestrator returns the SUM, and Parties divide by N after decryption.
        
        # For the purpose of this playground and the NoEncryption demo:
        # We will attempt to divide if it's a numpy array.
        # If it's an opaque object (Ciphertext), we might need to delegate or skip.
        
        # Let's assume for NoEncryption we return the true Mean.
        if isinstance(summed_model, np.ndarray):
            return summed_model / len(encrypted_models)
            
        # If we can't divide here (e.g. FHE ciphertext), we return the sum.
        # The Party will need to know to divide by N after decryption.
        # This is a bit of a leaky abstraction but necessary for partial homomorphic encryption.
        return summed_model
