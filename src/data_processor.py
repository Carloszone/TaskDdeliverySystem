import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Any, Dict, List, Union # Added Union for type hinting

class DataProcessor:
    """
    Processes raw observation data from the environment, preparing it for model input.
    This includes encoding, scaling, and potentially generating graph-like structures.
    """
    def __init__(self, num_categorical_features: int = 0, num_numerical_features: int = 0):
        """
        Initializes the DataProcessor.

        Args:
            num_categorical_features: Number of categorical features expected in the observation.
                                     Used to initialize LabelEncoders.
            num_numerical_features: Number of numerical features expected for scaling.
        """
        # Placeholder for coder: could be a list of LabelEncoders for categorical features
        # For simplicity, we assume categorical features are handled elsewhere or not present in the raw obs for now
        self.coder: List[LabelEncoder] = [LabelEncoder() for _ in range(num_categorical_features)]
        
        # Placeholder for scaler: StandardScaler for numerical features
        # If num_numerical_features is 0, scaler won't be fitted/used.
        self.scaler: StandardScaler = StandardScaler() if num_numerical_features > 0 else None
        
        # Placeholder for mapping: could store mappings from raw values to processed values or metadata
        self.mapping: Dict[str, Any] = {"example_mapping_key": "example_mapping_value"}

        self.num_categorical_features = num_categorical_features
        self.num_numerical_features = num_numerical_features
        self.fitted = False # To ensure scaler is fitted before transforming

        print(f"DataProcessor initialized. Categorical features: {num_categorical_features}, Numerical features: {num_numerical_features}")

    def fit_scaler(self, numerical_data_samples: np.ndarray):
        """
        Fits the scaler to sample numerical data.
        This should be called before using observation_processor if scaling is desired.

        Args:
            numerical_data_samples: A NumPy array where each row is a sample of numerical features.
                                   Shape should be (num_samples, num_numerical_features).
        """
        if self.scaler and numerical_data_samples.ndim == 2 and numerical_data_samples.shape[1] == self.num_numerical_features:
            print(f"Fitting StandardScaler with data of shape {numerical_data_samples.shape}...")
            self.scaler.fit(numerical_data_samples)
            self.fitted = True
            print("StandardScaler fitted.")
        elif self.scaler:
            print(f"Warning: Numerical data shape mismatch. Expected num_numerical_features={self.num_numerical_features}, got shape {numerical_data_samples.shape}. Scaler not fitted.")
        else:
            print("No scaler to fit (num_numerical_features is 0).")


    def observation_processor(self, observation: Union[List, np.ndarray]) -> np.ndarray:
        """
        Processes the raw observation from the environment.
        Applies encoding to categorical features and scaling to numerical features if configured.

        Args:
            observation: Raw observation data. Expected to be a list or NumPy array.
                         If configured with categorical/numerical features, it assumes a certain structure.
                         For this example, let's assume all features are numerical if scaler is active.

        Returns:
            A NumPy array representing the processed observation.
        """
        print(f"Original observation: {observation}")

        # Ensure observation is a NumPy array for processing
        if not isinstance(observation, np.ndarray):
            processed_observation = np.array(observation, dtype=np.float32)
        else:
            processed_observation = observation.astype(np.float32) # Ensure float type for scaler

        # --- More sophisticated processing would go here ---
        # 1. Separate categorical and numerical features based on a defined schema.
        # 2. Apply self.coder (e.g., LabelEncoder) to categorical features.
        #    Example:
        #    if self.num_categorical_features > 0:
        #        categorical_part = processed_observation[:, :self.num_categorical_features]
        #        numerical_part = processed_observation[:, self.num_categorical_features:]
        #        encoded_categorical_part = np.array([self.coder[i].transform(categorical_part[:, i]) for i in range(self.num_categorical_features)]).T
        #        # This requires coders to be fitted beforehand.
        #
        # 3. Apply self.scaler (e.g., StandardScaler) to numerical features.
        if self.scaler and self.fitted:
            # Assuming the entire 'processed_observation' is numerical and matches num_numerical_features
            if processed_observation.ndim == 1: # Single observation
                if len(processed_observation) == self.num_numerical_features:
                    processed_observation = self.scaler.transform(processed_observation.reshape(1, -1)).flatten()
                else:
                    print(f"Warning: Observation length {len(processed_observation)} does not match num_numerical_features {self.num_numerical_features} for scaling. Skipping scaling.")
            elif processed_observation.ndim == 2: # Batch of observations
                if processed_observation.shape[1] == self.num_numerical_features:
                    processed_observation = self.scaler.transform(processed_observation)
                else:
                    print(f"Warning: Observation feature count {processed_observation.shape[1]} does not match num_numerical_features {self.num_numerical_features} for scaling. Skipping scaling.")
            else:
                 print(f"Warning: Observation ndim {processed_observation.ndim} not suitable for scaler. Skipping scaling.")

        elif self.scaler and not self.fitted:
            print("Warning: Scaler is initialized but not fitted. Numerical features will not be scaled.")
        
        # 4. Potentially combine processed categorical and numerical features.
        #    Example (if separated earlier):
        #    processed_observation = np.concatenate((encoded_categorical_part, scaled_numerical_part), axis=1)
        
        print(f"Processed observation: {processed_observation}")
        return processed_observation

    def GNN_generator(self, processed_observation: np.ndarray) -> Any:
        """
        Prepares data for input to a Graph Neural Network (GNN) or Transformer model.
        This could involve generating graph structures (e.g., adjacency matrices, edge lists)
        or creating embeddings based on the processed observation.

        The README mentions "工序图变量（W） → Transformer 编码 → V_W" and
        "Transformer 提取上下文", suggesting that this method would handle the
        transformation of relevant parts of the observation into a format suitable
        for such sequence or graph-based models.

        Args:
            processed_observation: The observation data after initial processing by
                                   `observation_processor`.

        Returns:
            A placeholder value (e.g., the input itself, or a dummy graph representation).
            In a real implementation, this would be structured data for GNN/Transformer.
        """
        print(f"GNN_generator input (processed_observation): {processed_observation}")

        # --- GNN/Transformer data preparation logic would go here ---
        # 1. Identify parts of the processed_observation relevant for graph/sequence modeling
        #    (e.g., machine states, task dependencies, spatial relationships).
        # 2. Construct graph features:
        #    - Node features (e.g., from machine states in processed_observation).
        #    - Edge list or adjacency matrix representing relationships (e.g., based on '工序图变量 W').
        # 3. Or, prepare sequences for Transformer input.

        # Placeholder: Return the processed observation as is.
        # A more complex placeholder could be a dictionary:
        # return {
        #     "node_features": processed_observation, # Assuming rows are nodes
        #     "adjacency_matrix": np.eye(len(processed_observation)) # Dummy self-connections
        # }
        gnn_data_representation = processed_observation 
        
        print(f"GNN_generator output (dummy): {gnn_data_representation}")
        return gnn_data_representation

if __name__ == '__main__':
    print("--- DataProcessor Example Usage ---")

    # Example: Environment provides observations with 5 numerical features per arm, 2 arms
    # So, raw observation might be a flat list of 10 numbers.
    num_features = 10 
    processor = DataProcessor(num_categorical_features=0, num_numerical_features=num_features)

    # Sample observations (e.g., from an environment)
    # Batch of 3 observations, each with 10 features
    sample_obs_batch = np.array([
        [1.0, 2.0, 0.0, 1.0, 5.0, 1.5, 2.5, 0.5, 0.0, 4.0],
        [1.2, 2.1, 0.1, 1.3, 5.2, 1.4, 2.7, 0.3, 0.1, 4.3],
        [0.9, 1.9, 0.0, 0.8, 4.8, 1.6, 2.3, 0.6, 0.2, 3.8]
    ], dtype=np.float32)
    
    # Fit the scaler (important step before processing observations that need scaling)
    processor.fit_scaler(sample_obs_batch)

    # Process a single raw observation (list)
    raw_obs_single = [1.1, 2.2, 0.0, 0.9, 5.1, 1.3, 2.6, 0.4, 0.1, 4.1]
    print(f"\nProcessing single observation (list): {raw_obs_single}")
    processed_single = processor.observation_processor(raw_obs_single)
    # Pass it to GNN_generator
    gnn_input_single = processor.GNN_generator(processed_single)

    # Process a batch of raw observations (NumPy array)
    print(f"\nProcessing batch of observations (NumPy array): \n{sample_obs_batch}")
    processed_batch = processor.observation_processor(sample_obs_batch)
    # Pass it to GNN_generator (here, processing the whole batch as if it's one graph's node features)
    gnn_input_batch = processor.GNN_generator(processed_batch)
    
    print("\n--- Example with no scaler (num_numerical_features = 0) ---")
    processor_no_scale = DataProcessor(num_numerical_features=0)
    raw_obs_2 = np.array([10.0, 20.0, 30.0])
    processor_no_scale.fit_scaler(np.array([[1.0]])) # Will print warning or do nothing
    processed_2 = processor_no_scale.observation_processor(raw_obs_2)
    # Scaler should not have been applied.
    assert np.array_equal(processed_2, raw_obs_2.astype(np.float32)), "Scaler was applied when it shouldn't have been"
    print("Processed observation without scaling is as expected.")

    print("\nDataProcessor example usage finished.")
