import pandas as pd

class FeatureEngineer:
    def __init__(self, data, max_features=11):
        """
        Initializes the FeatureEngineer with the raw data and sets a maximum feature limit.
        Arguments:
        - data: The initial DataFrame to be processed.
        - max_features: Maximum number of features to limit the feature engineering process.
        """
        self.data = data.reset_index(drop=True)  # Ensure index is not counted as a feature
        self.max_features = max_features

    def generate_lagged_features(self, lags=None, columns=None):
        lags = lags or [1, 3, 5]
        columns = columns or ['Close', 'MSFT-Close']

        print("Generating lagged features...")
        for col in columns:
            for lag in lags:
                self.data[f'{col}_lag_{lag}'] = self.data[col].shift(lag)
        self.data.dropna(inplace=True)
        print(f"Generated lagged features with columns: {columns} and lags: {lags}")
        return self

    def add_rolling_features(self, window_sizes=None, columns=None):
        window_sizes = window_sizes or [7, 14]
        columns = columns or ['MSFT-Close']

        print("Adding rolling features...")
        for col in columns:
            for window in window_sizes:
                self.data[f'{col}_rolling_mean_{window}'] = self.data[col].rolling(window=window).mean()
                self.data[f'{col}_rolling_std_{window}'] = self.data[col].rolling(window=window).std()
        self.data.dropna(inplace=True)
        print(f"Added rolling features for columns: {columns} with window sizes: {window_sizes}")
        return self

    def add_technical_indicators(self):
        print("Adding technical indicators...")

        # RSI Calculation
        if 'RSI_y' in self.data.columns:
            self.data['RSI'] = self.data['RSI_y']
            print("Used existing RSI column: 'RSI_y'")
        else:
            window_length = 14
            delta = self.data['MSFT-Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=window_length).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=window_length).mean()
            rs = gain / loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            print("Calculated RSI from 'MSFT-Close'")

        # MACD Calculation
        ema_12 = self.data['MSFT-Close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.data['MSFT-Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = ema_12 - ema_26
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']
        print("Calculated MACD, MACD_signal, and MACD_histogram from 'MSFT-Close'")

        self.data.dropna(inplace=True)
        return self

    def finalize_features(self):
        """
        Finalizes the feature set, checking feature count and ensuring it does not exceed max_features.
        - Prints each feature name, index, and the final feature count.
        - Trims excess features if necessary.
        """
        # Only select columns that are actual features, excluding the target
        feature_columns = [col for col in self.data.columns if col != 'option_change_target']

        # Print columns for verification before trimming
        print("\nFeature columns before trimming:")
        for idx, col in enumerate(feature_columns):
            print(f"Index: {idx}, Column: {col}")

        # Trimming features if they exceed max_features
        if len(feature_columns) > self.max_features:
            print(f"\nReducing features from {len(feature_columns)} to {self.max_features}")
            selected_features = feature_columns[:self.max_features] + ['option_change_target']
            self.data = self.data[selected_features]
        else:
            print(f"\nFinal feature count within limit: {len(feature_columns)}")

        # Print final feature columns after trimming
        print("\nFinal feature columns after trimming:")
        for idx, col in enumerate(self.data.columns):
            print(f"Index: {idx}, Column: {col}")

        return self.data

    def run_feature_engineering(self):
        """
        Runs the complete feature engineering pipeline and returns the processed data.
        """
        return (self.generate_lagged_features()
                .add_rolling_features()
                .add_technical_indicators()
                .finalize_features())
