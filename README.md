# Federated Learning Playground with Encryption Support

This project provides a flexible playground for experimenting with Federated Learning (FL), specifically focusing on scenarios involving model encryption (e.g., Homomorphic Encryption).

## Features

- **Modular Design**: Separate components for Party, Orchestrator, Model, Encryption, Aggregation, and Data Loading.
- **Extensible**: easy to add new models, encryption schemes, or aggregation strategies.
- **Simulation**: `Environment` class to easily set up and run FL simulations.
- **Flexible Data**: Use synthetic data or load your own via CSV.
- **Analytics**: Tracks local and global model performance over rounds.

## Structure

```
fed_playground/
    src/
        encryption.py   # EncryptionScheme interface and NoEncryption
        models.py       # Model interface and LinearRegressionModel
        party.py        # Party class (local training)
        orchestrator.py # Orchestrator class (aggregation)
        environment.py  # Environment class (simulation loop)
        aggregation.py  # AggregationStrategy interface and MeanAggregation
        utils_data.py   # Data generation and splitting helpers
        dataloader.py   # DataLoader class for external files
    __init__.py
main.py                 # Example usage script
```

## Usage

### Using Synthetic Data
```bash
python main.py
```

### Using Custom Data (CSV)
1. Prepare a CSV file (e.g., `data.csv`) with a target column.
2. Initialize `DataLoader`:
```python
from fed_playground import DataLoader
loader = DataLoader('data.csv', target_column='target')
```
3. Pass it to `Environment`:
```python
env = Environment(..., data_loader=loader)
```

## Extending

### Adding a new Encryption Scheme

Inherit from `EncryptionScheme` in `src/encryption.py`:

```python
class MyFHEScheme(EncryptionScheme):
    def encrypt(self, params):
        # ...
    def decrypt(self, encrypted_params):
        # ...
    def aggregate(self, encrypted_params_list):
        # Perform homomorphic aggregation (e.g., sum)
```

### Adding a new Model

Inherit from `Model` in `src/models.py`.

### Adding a new Aggregation Strategy

Inherit from `AggregationStrategy` in `src/aggregation.py`.

## Requirements

- `numpy`
- `pandas`

## Example Output

```text
Round 1/10 - Avg Party Loss: 1.2345, Global Test Loss: 1.1000
...
Round 10/10 - Avg Party Loss: 0.0123, Global Test Loss: 0.0110
```
