import argparse
from modules import preprocess, train_evaluate

def main(args):
    # Preprocess the data
    preprocess.main(args.input_path, args.preprocess_output_dir)

    # The output of preprocessing becomes the input for training and evaluating
    preprocessed_data_path = f"{args.preprocess_output_dir}/preprocessed_data.csv"

    # Train and evaluate the model
    train_evaluate.main(preprocessed_data_path, args.train_output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loan Data Processing, Training, and Evaluation')
    parser.add_argument('--input_path', type=str, default='data/loan_data.csv')
    parser.add_argument('--preprocess_output_dir', type=str, default='data')
    parser.add_argument('--train_output_dir', type=str, default='analysis_outputs')
    args = parser.parse_args()

    main(args)

