from typing import Any
import numpy as np
from .models import Model
from .encryption import EncryptionScheme

class Party:
    """
    Represents a participant in the federated learning process.
    """
    def __init__(
        self, 
        party_id: int, 
        model: Model, 
        data: tuple[np.ndarray, np.ndarray], 
        encryption_scheme: EncryptionScheme
    ):
        self.party_id = party_id
        self.model = model
        self.X_train, self.y_train = data
        self.encryption_scheme = encryption_scheme

    def train_local_model(self) -> None:
        """
        Trains the local model on private data.
        """
        self.model.train(self.X_train, self.y_train)

    def get_encrypted_model(self) -> Any:
        """
        Returns the encrypted model parameters.
        """
        params = self.model.get_parameters()
        return self.encryption_scheme.encrypt(params)

    def update_model(self, global_model_params: np.ndarray) -> None:
        """
        Updates the local model with the global model parameters.
        Note: The global model parameters are expected to be decrypted (plaintext).
        In a real FHE setting, the party might receive an encrypted model and decrypt it,
        or the Orchestrator might have facilitated decryption (e.g., via MPC).
        Here we assume the Orchestrator sends back a plaintext model for simplicity/playground purposes,
        OR the party decrypts it if it was sent encrypted.
        
        For this interface, let's assume 'global_model_params' is what the party receives.
        If it's encrypted, we decrypt it.
        """
        # Check if the params are seemingly encrypted (this check is scheme dependent)
        # For our NoEncryption, they are just arrays. 
        # For this playground, let's explicitely use the scheme to decrypt if needed.
        # But usually Orchestrator sends back the aggregated result which might need decryption.
        
        # We will assume the input here is capable of being processed by 'decrypt' or set directly.
        # However, to be robust: let's try to decrypt. If the scheme says "I can't decrypt this" or it's already plain...
        # Actually, let's assume the interface `update_model` receives the result from the Orchestrator.
        # We invoke our scheme's decrypt.
        try:
            params = self.encryption_scheme.decrypt(global_model_params)
        except Exception:
            # Fallback or assume it was already plaintext if the scheme allows
            params = global_model_params
            
        self.model.set_parameters(params)

    def evaluate(self, X: np.ndarray = None, y: np.ndarray = None) -> float:
        """
        Evaluates the model.
        If no data is provided, evaluates on local training data.
        """
        if X is None or y is None:
            X, y = self.X_train, self.y_train
        return self.model.evaluate(X, y)
