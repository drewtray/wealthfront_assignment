import os
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def load_preprocessed_data(file_path):
    return pd.read_csv(file_path)

def train_evaluate(preprocessed_data_path, output_dir):
    '''
    """
    This function performs the following steps:
    1. Loads the preprocessed loan data from a CSV file.
    2. Specifies the feature and target columns.
    3. Splits the data into training and test sets.
    4. Defines the preprocessing pipeline for numeric and categorical features.
    5. Creates a pipeline that combines the preprocessor with a RandomForestClassifier.
    6. Defines the parameter grid for hyperparameter tuning.
    7. Performs the grid search with five-fold cross-validation.
    8. Prints the best parameters and the corresponding mean cross-validated score.
    9. Evaluates the best model on the test set and prints the classification report.
    10. Plot and save confusion matrix
    11. Save model weights
    
    Parameters:
    preprocessed_data_path (str): The file path to the CSV file containing preprocessed loan data.
    
    Returns:
    None
    """
    '''
    print("Training and evaluating the model...")
    
     # Step 1: Load the preprocessed loan data from a CSV file
    data = load_preprocessed_data(preprocessed_data_path)

    # Step 2: Specify the feature and target columns
    target = 'loan_status'  
    features = data.drop(target, axis=1)
    labels = data[target]

    # Step 3: Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Step 4: Define the preprocessing pipeline
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Step 5: Create a pipeline that combines the preprocessor with a RandomForestClassifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Step 6: Define the parameter grid for hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [100, 150],  
        'classifier__max_depth': [None, 10],  
        'classifier__min_samples_split': [2, 5],  
        'classifier__min_samples_leaf': [1, 2]  
    }

    # Step 7: Perform the grid search with five-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)   
    grid_search.fit(X_train, y_train)

    # Step 8: Print the best parameters and the corresponding mean cross-validated score
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validated score:", grid_search.best_score_)

    # Step 9: Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Step 10: Plot and save confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))  
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(ax=ax, cmap='Blues')  
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))  
    plt.close(fig)  

    # Step 11: Save the model weights
    model_weights_path = os.path.join(output_dir, 'model_weights.joblib')
    joblib.dump(best_model, model_weights_path)

def main(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_evaluate(input_path, output_dir)

if __name__ == '__main__':
    input_path = 'data/preprocessed_data.csv'  
    output_dir = 'analysis_outputs'  
    
    main(input_path, output_dir)