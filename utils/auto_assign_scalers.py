# Function to automatically classify variables into scalers based on statistical thresholds and predefined ordinal variables
def auto_assign_scalers(df, feat2scale, ordinal_vars=None):
    """
    Automatically assigns numerical variables to appropriate scalers based on statistical thresholds.
    
    Parameters:
        feat2scale: A list of numerical variables needed to be scaled.
        ordinal_vars (list): List of ordinal variables that should use MinMaxScaler.
        
    Returns:
        dict: Dictionary containing scaler assignments for each variable.
    """
    if ordinal_vars is None:
        ordinal_vars = []  # Default empty list if no ordinal variables specified
    
    scaler_groups = {"MinMaxScaler": [], "StandardScaler": [], "RobustScaler": [], "LogTransform": []}
    statistics = {'Scaler': [], 'Variable': [], 'Why': [], 'Outlier_count': [], 'Skewness': [] , 'Kurtosis': [], 'Range_val/IQR':[]}

    for col in feat2scale:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

        # Compute relevant statistics
        range_val = df[col].max() - df[col].min()
        median_distance = df[col].max() - df[col].median()
        outlier_count = len(outliers)
        skewness = df[col].skew()
        kurtosis = df[col].kurt()

        # Special handling for ordinal variables (force MinMaxScaler)
        if col in ordinal_vars:
            # For discrete ordinal variables, assign highly skewed ones for RobustScaler
            if outlier_count > 10:  
                scaler_groups["RobustScaler"].append(col)
                statistics['Scaler'].append("RobustScaler")
                statistics['Variable'].append(col)
                statistics['Outlier_count'].append(outlier_count)
                statistics['Skewness'].append(abs(skewness))
                statistics['Kurtosis'].append(abs(kurtosis))
                statistics['Range_val/IQR'].append(range_val/IQR  if IQR != 0 else np.nan)
                statistics['Why'].append('ordinal variable with outlier > 10')
            # For discrete ordinal variables, remaining for MinMaxScaler
            else:
                scaler_groups["MinMaxScaler"].append(col)
                statistics['Scaler'].append("MinMaxScaler")
                statistics['Variable'].append(col)
                statistics['Outlier_count'].append(outlier_count)
                statistics['Skewness'].append(abs(skewness))
                statistics['Kurtosis'].append(abs(kurtosis))
                statistics['Range_val/IQR'].append(range_val/IQR  if IQR != 0 else np.nan)
                statistics['Why'].append('ordinal variables')
        else:
            # For remaining continuous variables, firstly, find out normal distribution and assign them under Standard scaler
            # StandardScaler only for near-normal distributions with limited outliers
            if outlier_count < 5 and abs(skewness) < 0.5 and abs(kurtosis) < 3.0 and range_val < 5 * IQR:
                scaler_groups["StandardScaler"].append(col)
                statistics['Scaler'].append("StandardScaler")
                statistics['Variable'].append(col)
                statistics['Outlier_count'].append(outlier_count)
                statistics['Skewness'].append(abs(skewness))
                statistics['Kurtosis'].append(abs(kurtosis))
                statistics['Range_val/IQR'].append(range_val/IQR  if IQR != 0 else np.nan)
                statistics['Why'].append('outlier < 5 & skew < 0.5 & kurtosis < 3 & range < 5 * IQR')
            # For remaining continuous variables, secondly, find out less normal distribution and assign them under MinMaxScaler
            # MinMaxScaler for non-Gaussian but no extreme outliers
            elif outlier_count < 5 and abs(skewness) < 0.5 and range_val < 10 * IQR:
                scaler_groups["MinMaxScaler"].append(col)
                statistics['Scaler'].append("MinMaxScaler")
                statistics['Variable'].append(col)
                statistics['Outlier_count'].append(outlier_count)
                statistics['Skewness'].append(abs(skewness))
                statistics['Kurtosis'].append(abs(kurtosis))
                statistics['Range_val/IQR'].append(range_val/IQR  if IQR != 0 else np.nan)
                statistics['Why'].append('outlier < 5 & skew < 0.5 & range < 10 * IQR')
            # For remaining continuous variables, thirdly, find out extreme skewed distribution and assign them under Log Transformation. As the degree of skewness increases, PowerTransformer generally performs better than RobustScaler.
            # Log Transform for highly skewed distributions with heavy tails
            elif abs(skewness) > 1.0 and kurtosis > 2.0:
                scaler_groups["LogTransform"].append(col)
                statistics['Scaler'].append("LogTransform")
                statistics['Variable'].append(col)
                statistics['Outlier_count'].append(outlier_count)
                statistics['Skewness'].append(abs(skewness))
                statistics['Kurtosis'].append(abs(kurtosis))
                statistics['Range_val/IQR'].append(range_val/IQR if IQR != 0 else np.nan)
                statistics['Why'].append('skew > 1.0 & kurtosis > 2.0') 
            # For remaining continuous variables, fourthly, find out less extreme skewed distribution and assign them under RobustScaler
            # RobustScaler for strong outliers, large spread, or moderate skew
            elif outlier_count > 5 or range_val > 10 * IQR or abs(skewness) > 0.5:
                scaler_groups["RobustScaler"].append(col)
                statistics['Scaler'].append("RobustScaler")
                statistics['Variable'].append(col)
                statistics['Outlier_count'].append(outlier_count)
                statistics['Skewness'].append(abs(skewness))
                statistics['Kurtosis'].append(abs(kurtosis))
                statistics['Range_val/IQR'].append(range_val/IQR  if IQR != 0 else np.nan)
                statistics['Why'].append('outlier > 5 or skew > 0.5  or range > 10 * IQR')
            # For remaining continuous variables, at last, assign the remaining variables for MinMaxScaler
            else:
                scaler_groups["MinMaxScaler"].append(col)  # Default to StandardScaler
                statistics['Scaler'].append("MinMaxScaler")
                statistics['Variable'].append(col)
                statistics['Outlier_count'].append(outlier_count)
                statistics['Skewness'].append(abs(skewness))
                statistics['Kurtosis'].append(abs(kurtosis))
                statistics['Range_val/IQR'].append(range_val/IQR  if IQR != 0 else np.nan)
                statistics['Why'].append('Default as MinMaxScaler')  

    return scaler_groups,  statistics