import pandas as pd
import os

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def preprocess_data(input_path):
    '''
    Preprocess the loan data.

    Steps:
    1. Drop unnecessary columns
    2. Drop rows where only 'loan_amnt', 'funded_amnt', and 'addr_state' are non-null
    3. Drop the 'In-Grace' period records
    4. Combine loan statuses into "good" and "bad" categories
    5. Convert the employment length to numeric values
    6. Handle missing values (NaNs) for no employment history
    7. Fill NaNs with zeroes in 'mths_since_last_delinq'
    8. Replace '36 months' with 0 and '60 months' with numeric values
    9. Group rare categories into 'OTHER'
    10. Convert 'int_rate2' to float 
    

    Parameters:
    input_path (str): The file path to the CSV file containing loan data.

    Returns:
    pandas.DataFrame: Preprocessed loan data.
    '''
    data = load_data(input_path)
    if data is None:
        return None

     # Step 1: Drop unnecessary columns
    drop_columns = ['id', 'wtd_loans', 'interest_rate', 'num_rate', 'numrate', 'earliest_cr_line']
    data.drop(columns=drop_columns, inplace=True)

    # Step 2: Drop rows where only 'loan_amnt', 'funded_amnt', and 'addr_state' are non-null
    other_columns = [col for col in data.columns if col not in ['loan_amnt', 'funded_amnt', 'addr_state']]
    data = data.dropna(subset=other_columns, how='all')

    # Step 3: Drop the 'In-Grace' period records
    data = data[data['loan_status'] != 'In Grace Period']

    # Step 4: Combine loan statuses into "good" and "bad" categories
    # Here, assuming "Current" and "Fully Paid" are good, and all else are bad.
    status_mapping = {'Current': 1, 'Fully Paid': 1}
    data['loan_status'] = data['loan_status'].map(status_mapping).fillna(0)

    # Step 5: Convert the employment length to numeric values
    # Assume '< 1 year' as 0 and '10+ years' as 10
    mapping = {
        '< 1 year': 0,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10
    }
    data['emp_length'] = data['emp_length'].replace(mapping)

    # Step 6: Handle missing values (NaNs) for no employment history
    data['emp_length'].fillna(0, inplace=True)

    # Step 7: Fill NaNs with zeroes in 'mths_since_last_delinq'
    data['mths_since_last_delinq'].fillna(0, inplace=True)

    # Step 8: Replace '36 months' with 0 and '60 months' with numeric values
    data['term'] = data['term'].str.extract('(\d+)').astype(int)

    # Step 9: Group rare categories into 'OTHER'
    data.loc[~data['home_ownership'].isin(['MORTGAGE', 'RENT']), 'home_ownership'] = 'OTHER'

    # Step 10: Convert 'int_rate2' to float
    data['int_rate2'] = data['int_rate2'].str.rstrip('%').astype('float') / 100
    
    print("Data preprocessing completed.")
        
    return data

def main(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = 'preprocessed_data.csv'
    output_path = os.path.join(output_dir, output_file_name)
    
    # Process the data and save it to the output path
    processed_data = preprocess_data(input_path)
    if processed_data is not None:
        processed_data.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_path = 'data/loan_data.csv'  
    output_dir = 'data' 
    
    main(input_path, output_dir)