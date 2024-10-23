from typing import Dict, List

import pandas as pd
import re
import polyline
import numpy as np


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    """
    Reverses the input list by groups of n elements.
    """
    for i in range(0, len(lst), n):
        # Create a temporary list to hold the current group
        temp = []
        
        # Collect the current group (up to n elements)
        for j in range(i, min(i + n, len(lst))):
            temp.append(lst[j])
        
        # Reverse the temporary list manually
        reversed_temp = []
        for k in range(len(temp) - 1, -1, -1):
            reversed_temp.append(temp[k])
        
        # Extend the result with the reversed group
        result.extend(reversed_temp)
    
    return result

# Example usage:
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
n = 3
print(reverse_by_n_elements(my_list, n))


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}

    # Group strings by their lengths
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)

    # Sort the dictionary by keys (lengths)
    sorted_length_dict = dict(sorted(length_dict.items()))

    return sorted_length_dict

# Example usage:
string_list = ["apple", "banana", "kiwi", "pear", "grape", "fig", "orange"]
result = group_by_length(string_list)
print(result)

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    items = {}

    def flatten(current_dict: Dict, parent_key: str = ''):
        for key, value in current_dict.items():
            # Create the new key with the parent key and current key
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                # If the value is a dictionary, recurse into it
                flatten(value, new_key)
            elif isinstance(value, list):
                # If the value is a list, iterate over the elements
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        # If the list element is a dictionary, recurse into it
                        flatten(item, f"{new_key}[{i}]")
                    else:
                        # Otherwise, add the item directly
                        items[f"{new_key}[{i}]"] = item
            else:
                # For any other type, add it directly
                items[new_key] = value

    # Start flattening the nested dictionary
    flatten(nested_dict)
    return items

# Example usage:
nested_dict = {
    'name': 'John',
    'age': 30,
    'address': {
        'city': 'New York',
        'zipcode': '10001'
    },
    'phones': ['123-456-7890', '987-654-3210'],
    'children': [
        {
            'name': 'Alice',
            'age': 5
        },
        {
            'name': 'Bob',
            'age': 3
        }
    ]
}

flattened = flatten_dict(nested_dict)
print(flattened)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  # Add a copy of the current permutation
            return
        
        seen = set()  # To track duplicates at the current level
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue  # Skip duplicates
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]  # Swap
            backtrack(start + 1)  # Recur
            nums[start], nums[i] = nums[i], nums[start]  # Backtrack (swap back)

    nums.sort()  # Sort to handle duplicates
    result = []
    backtrack(0)
    return result

# Example usage:
input_list = [1, 1, 2]
unique_perms = unique_permutations(input_list)
print(unique_perms)


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Define a regex pattern for the three date formats
    date_pattern = r'''
        (?<!\d)          # Negative lookbehind to avoid matching part of larger numbers
        (                # Start of capturing group
            (?:         # Non-capturing group for different formats
                \d{2}  # Day (dd)
                -       # Separator
                \d{2}  # Month (mm)
                -       # Separator
                \d{4}  # Year (yyyy)
            |           # OR
                \d{2}  # Month (mm)
                /       # Separator
                \d{2}  # Day (dd)
                /       # Separator
                \d{4}  # Year (yyyy)
            |           # OR
                \d{4}  # Year (yyyy)
                \.      # Separator
                \d{2}  # Month (mm)
                \.      # Separator
                \d{2}  # Day (dd)
            )
        )                # End of capturing group
        (?!\d)          # Negative lookahead to avoid matching part of larger numbers
    '''

    # Find all matches in the input text
    matches = re.findall(date_pattern, text, re.VERBOSE)
    return matches

# Example usage:
text = "Here are some dates: 12-05-2023, 05/12/2023, 2023.05.12, invalid: 13-32-2023, and 2023.13.01."
found_dates = find_all_dates(text)
print(found_dates)

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Calculate the distance between successive points
    df['distance'] = 0.0  # Initialize distance column
    for i in range(1, len(df)):
        df.at[i, 'distance'] = haversine(
            (df.at[i-1, 'latitude'], df.at[i-1, 'longitude']),
            (df.at[i, 'latitude'], df.at[i, 'longitude'])
        )
    
    return df
def haversine(coord1, coord2):
    # Haversine formula to calculate the distance between two points on the Earth
    R = 6371000  # Radius of the Earth in meters
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c
# Example usage:
polyline_str = "u{~vHc{~n@R@G?@C?@A?@"
df = polyline_to_dataframe(polyline_str)
print(df)


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Create a transformed matrix with the new values
    transformed_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            transformed_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]

    return transformed_matrix

# Example usage:
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

result = rotate_and_multiply_matrix(matrix)
print(result)


def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Convert date and time columns to datetime
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Create a multi-index from id and id_2
    df.set_index(['id', 'id_2'], inplace=True)

    # Group by id and id_2
    grouped = df.groupby(level=['id', 'id_2'])

    # Initialize a boolean Series for results
    completeness_check = pd.Series(index=grouped.groups.keys(), dtype=bool)

    # Check each group for time completeness
    for (id_val, id_2_val), group in grouped:
        # Check if it covers all 7 days of the week
        days_covered = group['start_datetime'].dt.day_name().unique()
        full_week = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        # Check if start and end times span the full 24 hours
        starts = group['start_datetime'].min()
        ends = group['end_datetime'].max()
        full_day_covered = (starts.time() == pd.Timestamp('00:00:00').time() and 
                            ends.time() == pd.Timestamp('23:59:59').time())

        # Update the completeness check for this (id, id_2) pair
        completeness_check[(id_val, id_2_val)] = not (full_week.issubset(days_covered) and full_day_covered)

    return completeness_check

# Example usage:

file_path = 'E:/path/to/your/dataset-1.csv'
df = pd.read_csv(file_path)

# Now call the function
result = time_check(df)
print(result)