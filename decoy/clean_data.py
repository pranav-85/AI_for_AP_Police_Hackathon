#Remove duplicate rows
import pandas as pd

def clean_data(file_path):
    # Load the data
    df = pd.read_excel(file_path)

    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Save the cleaned data back to Excel
    cleaned_file_path = file_path.replace(".xlsx", "_cleaned.xlsx")
    df_cleaned.to_excel(cleaned_file_path, index=False)
    
    print(f"Cleaned data saved to '{cleaned_file_path}'.")

if __name__ == "__main__":
    file_path = "test.xlsx"
    clean_data(file_path)