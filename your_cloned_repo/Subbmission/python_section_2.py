import pandas as pd
import numpy as np
from datetime import time


def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Assume the DataFrame has columns: 'id_from', 'id_to', 'distance'
    
    unique_ids = pd.concat([df['id_from'], df['id_to']]).unique()
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    for _, row in df.iterrows():
        id_from = row['id_from']
        id_to = row['id_to']
        distance = row['distance']
        
        distance_matrix.at[id_from, id_to] = distance
        distance_matrix.at[id_to, id_from] = distance

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, k] + distance_matrix.at[k, j] < distance_matrix.at[i, j] or distance_matrix.at[i, j] == 0:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix

# Example usage with the full path
df = pd.read_csv('C:/path/to/your/dataset-2.csv')  # Adjust this path
distance_matrix = calculate_distance_matrix(df)
print(distance_matrix)



def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Reset the index to use IDs
    unrolled = df.reset_index()
    
    # Melt the DataFrame to long format
    long_format = unrolled.melt(id_vars='index', var_name='id_end', value_name='distance')
    
    # Rename 'index' to 'id_start'
    long_format.rename(columns={'index': 'id_start'}, inplace=True)
    
    # Filter out rows where id_start is the same as id_end
    long_format = long_format[long_format['id_start'] != long_format['id_end']]
    
    return long_format
def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Assume the DataFrame has columns: 'id_from', 'id_to', 'distance'
    
    unique_ids = pd.concat([df['id_from'], df['id_to']]).unique()
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    for _, row in df.iterrows():
        id_from = row['id_from']
        id_to = row['id_to']
        distance = row['distance']
        
        distance_matrix.at[id_from, id_to] = distance
        distance_matrix.at[id_to, id_from] = distance

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, k] + distance_matrix.at[k, j] < distance_matrix.at[i, j] or distance_matrix.at[i, j] == 0:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix
# Example usage
# Load your dataset
df = pd.read_csv('dataset-2.csv')  # Ensure the correct path to your CSV file

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(df)

# Unroll the distance matrix
unrolled_df = unroll_distance_matrix(distance_matrix)

# Print the unrolled DataFrame
print(unrolled_df)

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> list:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Assume the DataFrame has columns: 'id_from', 'id_to', 'distance'
    unique_ids = pd.concat([df['id_from'], df['id_to']]).unique()
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    for _, row in df.iterrows():
        id_from = row['id_from']
        id_to = row['id_to']
        distance = row['distance']
        
        distance_matrix.at[id_from, id_to] = distance
        distance_matrix.at[id_to, id_from] = distance

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, k] + distance_matrix.at[k, j] < distance_matrix.at[i, j] or distance_matrix.at[i, j] == 0:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Reset the index to use IDs
    unrolled = df.reset_index()
    
    # Melt the DataFrame to long format
    long_format = unrolled.melt(id_vars='index', var_name='id_end', value_name='distance')
    
    # Rename 'index' to 'id_start'
    long_format.rename(columns={'index': 'id_start'}, inplace=True)
    
    # Filter out rows where id_start is the same as id_end
    long_format = long_format[long_format['id_start'] != long_format['id_end']]
    
    return long_format

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> list:
    # Filter the DataFrame for the given reference ID
    avg_distance = df.loc[df['id_start'] == reference_id, 'distance'].mean()

    if pd.isna(avg_distance):
        return []  # If the reference ID does not exist, return an empty list

    # Calculate the thresholds
    lower_threshold = avg_distance * 0.9
    upper_threshold = avg_distance * 1.1

    # Find IDs within the 10% threshold
    filtered_ids = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]
    
    # Get unique id_start values and sort them
    unique_ids = sorted(filtered_ids['id_start'].unique())
    
    return unique_ids

# Example usage
df = pd.read_csv('dataset-2.csv')  # Ensure the correct path to your CSV file

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(df)

# Unroll the distance matrix
unrolled_df = unroll_distance_matrix(distance_matrix)

# Set your reference ID
reference_id = ...  # Replace with a specific ID you want to check

# Find IDs within the 10% threshold
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)

# Print the result
print(ids_within_threshold)


def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define the rate coefficients
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates and add new columns
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate
    
    return df
def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    unique_ids = pd.concat([df['id_from'], df['id_to']]).unique()
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    for _, row in df.iterrows():
        id_from = row['id_from']
        id_to = row['id_to']
        distance = row['distance']
        
        distance_matrix.at[id_from, id_to] = distance
        distance_matrix.at[id_to, id_from] = distance

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, k] + distance_matrix.at[k, j] < distance_matrix.at[i, j] or distance_matrix.at[i, j] == 0:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    unrolled = df.reset_index()
    long_format = unrolled.melt(id_vars='index', var_name='id_end', value_name='distance')
    long_format.rename(columns={'index': 'id_start'}, inplace=True)
    long_format = long_format[long_format['id_start'] != long_format['id_end']]
    return long_format
# Example usage
df = pd.read_csv('dataset-2.csv')  # Ensure the correct path to your CSV file

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(df)

# Unroll the distance matrix
unrolled_df = unroll_distance_matrix(distance_matrix)

# Calculate toll rates
toll_rate_df = calculate_toll_rate(unrolled_df)

# Print the resulting DataFrame
print(toll_rate_df)


def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    weekday_discount_factors = {
        (time(0, 0), time(10, 0)): 0.8,
        (time(10, 0), time(18, 0)): 1.2,
        (time(18, 0), time(23, 59, 59)): 0.8
    }
    weekend_discount_factor = 0.7
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Adding columns with dummy data for demonstration
    df['start_day'] = np.random.choice(days_of_week, size=len(df))
    df['end_day'] = df['start_day']
    df['start_time'] = time(0, 0)
    df['end_time'] = time(23, 59, 59)

    for index, row in df.iterrows():
        start_day = row['start_day']
        distance = row['distance']
        
        if start_day in ['Saturday', 'Sunday']:
            df.at[index, 'moto'] *= weekend_discount_factor
            df.at[index, 'car'] *= weekend_discount_factor
            df.at[index, 'rv'] *= weekend_discount_factor
            df.at[index, 'bus'] *= weekend_discount_factor
            df.at[index, 'truck'] *= weekend_discount_factor
        else:
            for time_range, factor in weekday_discount_factors.items():
                if row['start_time'] >= time_range[0] and row['start_time'] < time_range[1]:
                    df.at[index, 'moto'] *= factor
                    df.at[index, 'car'] *= factor
                    df.at[index, 'rv'] *= factor
                    df.at[index, 'bus'] *= factor
                    df.at[index, 'truck'] *= factor
                    break

    return df
def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    unique_ids = pd.concat([df['id_from'], df['id_to']]).unique()
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    for _, row in df.iterrows():
        id_from = row['id_from']
        id_to = row['id_to']
        distance = row['distance']
        
        distance_matrix.at[id_from, id_to] = distance
        distance_matrix.at[id_to, id_from] = distance

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, k] + distance_matrix.at[k, j] < distance_matrix.at[i, j] or distance_matrix.at[i, j] == 0:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix
def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    unrolled = df.reset_index()
    long_format = unrolled.melt(id_vars='index', var_name='id_end', value_name='distance')
    long_format.rename(columns={'index': 'id_start'}, inplace=True)
    long_format = long_format[long_format['id_start'] != long_format['id_end']]
    return long_format
def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate
    
    return df
df = pd.read_csv('dataset-2.csv')  # Ensure the correct path to your CSV file
distance_matrix = calculate_distance_matrix(df)
unrolled_df = unroll_distance_matrix(distance_matrix)
toll_rate_df = calculate_toll_rate(unrolled_df)
time_based_toll_rates_df = calculate_time_based_toll_rates(toll_rate_df)

# Print the resulting DataFrame
print(time_based_toll_rates_df)
