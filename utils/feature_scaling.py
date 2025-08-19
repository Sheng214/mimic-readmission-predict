import numpy as np

def feature_scaling(X_train, X_test, cols_to_scaling, statistics):

    # Define the methods dictionary
    method_dict = {
        'reference_year': lambda df, col, ref_year: df[col] - ref_year,
        'sin_cos_transform': lambda df, col: (
            df.assign(**{f"{col}_sin": np.sin(2 * np.pi * df[col] / 12),
                        f"{col}_cos": np.cos(2 * np.pi * df[col] / 12)}).drop(columns=[col])
        ),
        'MinMaxScaler': lambda: MinMaxScaler(), # Factory function to create a new instance
        'StandardScaler': lambda: StandardScaler(),
        'RobustScaler': lambda: RobustScaler(),
        'LogTransform': lambda df, col: np.log1p(df[col])  # log(1 + X) to handle zero values
    }

    # Define a dictionary of each variable as key and its assigned scaler as value
    variable_to_method = dict(zip(statistics['Variable'], statistics['Scaler']))

    # Create a dictionary to store fitted scalers
    scaler_instances = {}

    year_cols = []  # Default empty list if no specified
        
    month_cols = []  # Default empty list if no specified

    # Apply scaler methods
    for col in cols_to_scaling:
        method = variable_to_method[col]
        # Create a new scaler instance for each column
        scaler = method_dict[method]() if method in ['MinMaxScaler', 'StandardScaler', 'RobustScaler'] else method_dict[method]

        if col in year_cols:
            # Subtract reference year
            ref_year = X_train[col].min()
            X_train[col] = scaler(X_train, col, ref_year)
            X_test[col] = scaler(X_test, col, ref_year)
            print(f'reference year for {col} is {ref_year}')
            
        elif col in month_cols:
            # Apply sin-cos transformation and update DataFrames
            X_train = scaler(X_train, col)
            X_test = scaler(X_test, col)
        
        
        elif method == 'LogTransform':
            # Apply log transformation
            X_train[col] = scaler(X_train, col)
            if col in X_test.columns:
                X_test[col] = scaler(X_test, col)

        else:
            # Fit on train, transform on both train and test
            scaler_instances[col] = scaler.fit(X_train[[col]])
            X_train[col] = scaler_instances[col].transform(X_train[[col]])
            X_test[col] = scaler_instances[col].transform(X_test[[col]])
    return X_train, X_test
