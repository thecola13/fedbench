from typing import List, Dict, Any
import numpy as np
from .aggregation import AggregationStrategy
from .encryption import EncryptionScheme

class Orchestrator:
    """
    Coordinator for the federated learning process.
    Aggregates models from parties.
    """
    def __init__(
        self, 
        aggregation_strategy: AggregationStrategy,
        encryption_scheme: EncryptionScheme,
        initial_model_params: np.ndarray = None
    ):
        self.aggregation_strategy = aggregation_strategy
        self.encryption_scheme = encryption_scheme
        self.global_model_params = initial_model_params
        self.parties: List = [] # Avoid circular import type hint

    def register_party(self, party) -> None:
        self.parties.append(party)

    def distribute_model(self) -> None:
        """
        Sends the current global model to all registered parties.
        """
        if self.global_model_params is None:
            return

        for party in self.parties:
            # Depending on the protocol, we might send Encrypted(Global) or Plain(Global).
            # For this playground, we distribute the parameter as is.
            # The Party.update_model method handles decryption if needed.
            party.update_model(self.global_model_params)

    def aggregate_models(self) -> None:
        """
        Collects encrypted models from parties and aggregates them.
        """
        encrypted_models = []
        for party in self.parties:
            encrypted_models.append(party.get_encrypted_model())
        
        aggregated_result = self.aggregation_strategy.aggregate(encrypted_models, self.encryption_scheme)
        
        # Update global model
        self.global_model_params = aggregated_result
