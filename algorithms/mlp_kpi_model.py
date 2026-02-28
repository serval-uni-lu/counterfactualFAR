"""
MLP Neural Network model that integrates MAKPIGenerator in the forward pass.
The model takes raw time series data as input and generates KPIs internally for prediction.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import random
from utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL
from algorithms.kpi_gen.ma_kpi_generator import MAKPIGenerator


class KPIGeneratorModule(nn.Module):
    """
    KPI generator module that produces KPI features from raw time series.
    """

    def __init__(self, k=5, kpi_type="full", kpi_features=None):
        super().__init__()
        self.k = k
        self.kpi_type = kpi_type
        self.kpi_features = kpi_features
        self.kpi_generator = MAKPIGenerator(data=None, k=k, kpi_type=kpi_type)

    def generate_kpis_df(self, time_series_df):
        self.kpi_generator.data = time_series_df
        self.kpi_generator.k = self.k
        self.kpi_generator.kpi_type = self.kpi_type
        self.kpi_generator.compute()
        return self.kpi_generator.get_kpis()

    def forward(self, time_series_df):
        if self.kpi_features is None:
            raise ValueError("kpi_features must be set before generating KPIs")

        kpis = self.generate_kpis_df(time_series_df)
        features = kpis[self.kpi_features].values
        return torch.FloatTensor(features)


class LowerBoundedOutput(nn.Module):
    def __init__(self, lower_bound=-5.0):
        super().__init__()
        self.lower_bound = float(lower_bound)
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x) + self.lower_bound


class MLPKPIModel(nn.Module):
    """
    MLP Neural Network model that integrates KPI generation internally.
    Takes raw time series data as input, generates KPIs in the forward pass, and predicts profitability.
    """

    def __init__(self, hidden_sizes=None, k=5, kpi_type="full", kpi_features=None, seed=42, target_lower_bound=-5.0):
        """
        Initialize the MLP model with internal KPI generation capability.
        
        :param hidden_sizes: List of hidden layer sizes. Default: [128, 64, 32]
        :param k: Parameter for moving average in KPI generation
        :param kpi_type: Type of KPIs to generate ("full", "basic", "short", "full_short")
        :param kpi_features: List of KPI feature names to use as input to the MLP
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.hidden_sizes = hidden_sizes
        self.k = k
        self.kpi_type = kpi_type
        self.kpi_features = kpi_features
        self.seed = seed
        self.target_lower_bound = float(target_lower_bound)
        self.is_fitted = False

        self.kpi_module = KPIGeneratorModule(k=k, kpi_type=kpi_type, kpi_features=kpi_features)

        self.input_size = None
        self.network = None
        self.device = "cpu"

    def _select_features_in_order(self, df):
        missing = [col for col in self.kpi_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing KPI features for MLP model: {missing}")
        return df[self.kpi_features]

    def build_network(self, input_size):
        """Build the network once we know the input size (number of KPI features)."""
        if self.network is not None:
            return  # Already built
        
        self.input_size = input_size
        
        # Build the neural network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        # Output layer: single prediction (profitability), constrained to [target_lower_bound, +inf)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(LowerBoundedOutput(self.target_lower_bound))
        
        self.network = nn.Sequential(*layers)

    def forward(self, time_series_df):
        """
        Forward pass through the network.
        Takes raw time series data, generates KPIs internally, then predicts.
        
        :param time_series_df: DataFrame with raw time series data
        :return: Prediction tensor
        """
        x = self.kpi_module(time_series_df)
        
        # Build network if not already built
        if self.network is None:
            self.build_network(x.shape[1])
        
        # Pass through the network
        output = self.network(x)
        return output

    def fit(self, time_series_df, y, kpi_features, epochs=300, batch_size=2048, learning_rate=3e-4,
        weight_decay=5e-4, val_split=0.15, patience=25, device='cpu', artifact_label=None):
        """
        Train the model.
        
        :param time_series_df: Raw time series data (DataFrame)
        :param y: Training targets (numpy array or Series)
        :param kpi_features: List of KPI feature names to use
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        :param learning_rate: Learning rate for optimizer
        :param device: Device to train on ('cpu' or 'cuda')
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Set the KPI features
        self.kpi_features = kpi_features
        self.kpi_module.kpi_features = kpi_features
        
        # Generate KPIs from the full time series via module
        kpis_df = self.kpi_module.generate_kpis_df(time_series_df)
        if artifact_label is not None:
            os.makedirs("for_testing", exist_ok=True)
            if isinstance(y, pd.DataFrame):
                target_cols = [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "target"]
                if all(col in y.columns for col in target_cols):
                    kpis_to_save = kpis_df.merge(
                        y[target_cols],
                        on=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
                        how="left",
                    )
                else:
                    kpis_to_save = kpis_df
            else:
                kpis_to_save = kpis_df
            kpis_to_save.to_csv(f"for_testing/generated_kpis_train_{artifact_label}_mlp.csv", index=False)

        if isinstance(y, pd.DataFrame):
            required_target_cols = [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "target"]
            missing_target_cols = [col for col in required_target_cols if col not in y.columns]
            if missing_target_cols:
                raise ValueError(
                    f"Missing required target columns for MLP training: {missing_target_cols}"
                )

            merged = kpis_df.merge(
                y[required_target_cols],
                on=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
                how="inner",
            )

            train_columns = list(self.kpi_features) + [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "target"]
            merged = merged[train_columns].dropna()

            X = self._select_features_in_order(merged).values
            y = merged["target"].values
        else:
            X = self._select_features_in_order(kpis_df).values

        # Convert to tensors
        if isinstance(y, pd.Series):
            y = y.values
        
        y = np.asarray(y).reshape(-1, 1)

        y_min = float(np.nanmin(y)) if y.size > 0 else float("inf")
        if y_min < self.target_lower_bound:
            raise ValueError(
                "Found training target(s) below the configured lower bound: "
                f"min_target={y_min:.6f}, lower_bound={self.target_lower_bound:.6f}. "
                "Either increase the lower bound flexibility (e.g., use a smaller lower bound) "
                "or clean/cap targets to match the model constraint."
            )

        if X.shape[0] < 10:
            val_split = 0.0

        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # Build network if not already built
        if self.network is None:
            self.build_network(X.shape[1])
        
        # Move model to device
        self.device = device
        self.to(device)
        
        # Create train/validation dataloaders
        dataset = torch.utils.data.TensorDataset(X, y)
        split_generator = torch.Generator().manual_seed(self.seed)
        if val_split > 0.0:
            val_size = max(1, int(len(dataset) * val_split))
            train_size = len(dataset) - val_size
            if train_size < 1:
                train_size = len(dataset)
                val_size = 0
            if val_size > 0:
                train_dataset, val_dataset = torch.utils.data.random_split(
                    dataset,
                    [train_size, val_size],
                    generator=split_generator,
                )
            else:
                train_dataset, val_dataset = dataset, None
        else:
            train_dataset, val_dataset = dataset, None

        loader_generator = torch.Generator().manual_seed(self.seed)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        val_loader = None if val_dataset is None else torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Define loss and optimizer
        criterion = nn.SmoothL1Loss(beta=0.5)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
        
        # Training loop
        best_val = float("inf")
        best_state = None
        wait = 0

        self.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                outputs = self.network(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()

            train_loss = train_loss / max(1, len(train_loader))

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        outputs = self.network(batch_X)
                        val_loss += criterion(outputs, batch_y).item()
                val_loss = val_loss / max(1, len(val_loader))
                scheduler.step(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
                    wait = 0
                else:
                    wait += 1

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")

                if wait >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                self.train()
            else:
                scheduler.step(train_loss)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

        if best_state is not None:
            self.network.load_state_dict(best_state)
        
        self.is_fitted = True
        self.eval()

    def predict(self, time_series_df):
        """
        Make predictions on new data.
        
        :param time_series_df: Raw time series data (DataFrame)
        :return: Predictions (numpy array)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.kpi_features is None:
            raise ValueError("kpi_features must be set (call fit() first)")
        
        X = self.kpi_module(time_series_df).numpy()
        X = torch.FloatTensor(X).to(self.device)
        
        self.eval()
        with torch.no_grad():
            outputs = self.network(X)
        
        return outputs.cpu().numpy()

    def predict_with_kpis(self, time_series_df, artifact_label=None):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if self.kpi_features is None:
            raise ValueError("kpi_features must be set (call fit() first)")

        kpis_df = self.kpi_module.generate_kpis_df(time_series_df)
    
        X = self._select_features_in_order(kpis_df).values
        X = torch.FloatTensor(X).to(self.device)

        self.eval()
        with torch.no_grad():
            outputs = self.network(X)

        return kpis_df, outputs.cpu().numpy()
