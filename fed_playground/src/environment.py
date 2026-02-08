from typing import Dict, List, Any, Optional
import numpy as np

from .models import Model, LinearRegressionModel
from .encryption import EncryptionScheme, NoEncryption
from .aggregation import AggregationStrategy, MeanAggregation
from .party import Party
from .orchestrator import Orchestrator
from .utils_data import generate_linear_data, split_data
from .dataloader import DataLoader

class Environment:
    """
    Sets up and runs the federated learning simulation.
    """
    def __init__(
        self,
        n_parties: int,
        encryption_scheme: EncryptionScheme,
        aggregation_strategy: AggregationStrategy,
        n_features: int = 0, # Optional if data_loader is used
        n_samples: int = 0,  # Optional if data_loader is used
        model_class: type = LinearRegressionModel,
        model_params: Dict[str, Any] = {},
        data_loader: Optional[DataLoader] = None
    ):
        self.n_parties = n_parties
        self.n_features = n_features
        self.n_samples = n_samples
        self.encryption_scheme = encryption_scheme
        self.aggregation_strategy = aggregation_strategy
        self.model_class = model_class
        self.model_params = model_params
        self.data_loader = data_loader
        
        self.parties: List[Party] = []
        self.orchestrator: Orchestrator = None
        
        # Analytics
        self.history = {
            'global_loss': [],
            'party_loss': []
        }
        
    def setup(self):
        """
        Initializes data, parties, and orchestrator.
        """
        # 1. Load or Generate Data
        if self.data_loader:
            X, y = self.data_loader.load()
            self.n_samples = X.shape[0]
            self.n_features = X.shape[1]
        else:
            X, y = generate_linear_data(self.n_samples, self.n_features)
        
        # 2. Split Data (Hold out some for global testing if needed, but for now use all)
        # Let's keep 20% for global evaluation
        split_idx = int(0.8 * self.n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.test_data = (X_test, y_test)
        
        party_data_splits = split_data(X_train, y_train, self.n_parties)
        
        # 3. Initialize Parties
        for i in range(self.n_parties):
            # Create a fresh model instance for each party
            model = self.model_class(input_dim=self.n_features, **self.model_params)
            party = Party(
                party_id=i,
                model=model,
                data=party_data_splits[i],
                encryption_scheme=self.encryption_scheme
            )
            self.parties.append(party)
            
        # 4. Initialize Orchestrator
        # Orchestrator needs to know the shape of the model to initialize global params (optional)
        # Or it can wait for the first round.
        # Let's initialize global params from a fresh model.
        initial_model = self.model_class(input_dim=self.n_features, **self.model_params)
        initial_params = initial_model.get_parameters()
        
        self.orchestrator = Orchestrator(
            aggregation_strategy=self.aggregation_strategy,
            encryption_scheme=self.encryption_scheme,
            initial_model_params=initial_params
        )
        
        # Register parties
        for party in self.parties:
            self.orchestrator.register_party(party)
            
    def run_round(self):
        """
        Executes one round of Federated Learning.
        """
        # 1. Orchestrator broadcasts global model
        self.orchestrator.distribute_model()
        
        # 2. Parties train locally
        avg_party_loss = 0
        for party in self.parties:
            party.train_local_model()
            avg_party_loss += party.evaluate() # Evaluate on train data
        avg_party_loss /= self.n_parties
        self.history['party_loss'].append(avg_party_loss)
        
        # 3. Orchestrator aggregates
        self.orchestrator.aggregate_models()
        
        # 4. Global Evaluation
        # We need a model instance to evaluate the global parameters
        global_params = self.orchestrator.global_model_params
        # If encrypted, this evaluation might fail or need decryption.
        # For NoEncryption, it's fine.
        
        # Check if we can decrypt for evaluation purposes (assuming Environment has 'god mode')
        if not isinstance(global_params, np.ndarray):
             try:
                 global_params = self.encryption_scheme.decrypt(global_params)
             except:
                 pass # Cannot evaluate encrypted model
        
        if isinstance(global_params, np.ndarray):
            eval_model = self.model_class(input_dim=self.n_features, **self.model_params)
            eval_model.set_parameters(global_params)
            X_test, y_test = self.test_data
            global_loss = eval_model.evaluate(X_test, y_test)
            self.history['global_loss'].append(global_loss)
        else:
            self.history['global_loss'].append(None)

    def run_simulation(self, rounds: int = 10, test_data: tuple = None):
        """
        Runs the federated learning simulation for the specified number of rounds.
        
        Args:
            rounds: Number of FL rounds to execute
            test_data: Optional tuple of (X_test, y_test) for global evaluation.
                      If not provided, uses the test set created during setup.
        
        Returns:
            dict: History containing 'global_loss' and 'party_loss' for each round
        """
        self.setup()
        
        # Override test data if provided
        if test_data is not None:
            self.test_data = test_data
        
        for r in range(rounds):
            self.run_round()
            print(f"Round {r+1}/{rounds} - "
                  f"Avg Party Loss: {self.history['party_loss'][-1]:.4f}, "
                  f"Global Test Loss: {self.history['global_loss'][-1]}")
        
        return self.history
