import os
import pandas as pd

# --- Docling Imports ---
try:
    from docling.document_converter import DocumentConverter
except ImportError as e:
    print(f"Error importing Docling components: {e}")
    print("Please install Docling: pip install 'docling[all]'")
    exit()

def parse_clinical_data(table_df: pd.DataFrame):
    """
    [FINAL, ROBUST VERSION]
    This parser first sanitizes the DataFrame's columns and then uses
    more reliable logic to split the data into sub-tables.
    """
    # 1. Sanitize column names to prevent errors from empty/duplicate names
    sanitized_columns = []
    for i, col in enumerate(table_df.columns):
        col_str = str(col).strip()
        if not col_str:
            sanitized_columns.append(f"unnamed_col_{i}")
        else:
            sanitized_columns.append(col_str)
    table_df.columns = sanitized_columns

    # Use the sanitized names for reliable access
    header_col = sanitized_columns[0]
    char_col = sanitized_columns[1]
    val_cols = sanitized_columns[2:]

    records = table_df.to_dict('records')
    extracted_tables = {}
    current_table_name = None
    current_table_rows = []
    final_column_names = []

    for row in records:
        section_title = str(row[header_col]).strip()

        # A new section title ALWAYS marks the start of a new table
        if section_title:
            # If we were building a previous table, save it now.
            if current_table_name and current_table_rows:
                df = pd.DataFrame(current_table_rows, columns=final_column_names)
                extracted_tables[current_table_name] = df

            # Initialize the new table
            current_table_name = section_title
            current_table_rows = []
            
            # The column headers for the final DataFrame are defined in this row
            final_column_names = [str(row[char_col]).strip() or "Characteristic"] + [str(row[c]) for c in val_cols]

        # If we are inside a table, this is a data row
        elif current_table_name:
            # A data row must have a value in the characteristic column
            characteristic = str(row[char_col]).strip()
            if characteristic:
                data_values = [characteristic] + [str(row[c]) for c in val_cols]
                current_table_rows.append(data_values)

    # After the loop, the very last table is still in memory. Save it.
    if current_table_name and current_table_rows:
        df = pd.DataFrame(current_table_rows, columns=final_column_names)
        extracted_tables[current_table_name] = df

    return extracted_tables


class TableQueryEngine:
    """
    This class is correct and requires no changes. It initializes the parser
    and provides the query interface.
    """
    def __init__(self, file_path: str):
        print(f"🔬 Initializing engine by processing document: {file_path}")
        self._tables = {}
        try:
            doc_converter = DocumentConverter()
            conv_result = doc_converter.convert(file_path)
            doc = conv_result.document
            
            if doc.tables:
                main_table_df = doc.tables[0].export_to_dataframe()
                self._tables = parse_clinical_data(main_table_df)
                print(f"✅ Engine Initialized. Found and parsed {len(self._tables)} sub-tables.")
            else:
                print("❌ No tables found by Docling.")
        except Exception as e:
            print(f"❌ Critical error during initialization: {e}")

    def query(self, query_string: str) -> pd.DataFrame | None:
        query_lower = query_string.lower()
        for table_name, df in self._tables.items():
            if query_lower in table_name.lower():
                return df
        print(f"⚠️ No table found matching query: '{query_string}'")
        return None
    
    def list_tables(self) -> list[str]:
        return list(self._tables.keys())


if __name__ == "__main__":
    file_path = "/Users/Sravya/Desktop/AI_model_table_shells/mark - can you convert this into a markdown.csv"

    # 1. Initialize the engine.
    engine = TableQueryEngine(file_path)
    
    print("\n📋 Available tables for querying:")
    for name in engine.list_tables():
        print(f"  - {name}")

    # 2. Define queries.
    queries = [
        "Ethnicity",
        "Height (cm)",
        "Weight (kg)",
        "Body Mass Index (kg/m2)",
        "Body Surface Area (m2)",
        "Stratification Factor 1 (EDC)",
        "Stratification Factor 1 (LxRS)",
        "ECOG Performance Status",
        "12-Lead ECG",
        "Baseline/Biomarker Subgroups",
        "Smoking Status",
        "Bone marrow/aspirate blast count at baseline",
        "Region/Country of Enrollment",
        "Time from Initial Histologic Diagnosis to Randomization (days)",
        "Histology",
        "Tumor Stage at Initial Diagnosis",
        "Tumor Stage at Study Entry",
        "TNM Stage at Study Entry (T)",
        "TNM Stage at Study Entry (N)"
    ]

    print("\n--- Running Queries ---")
    # 3. Loop and print results.
    for q in queries:
        print(f"\n▶️ Querying for: '{q}'")
        result_df = engine.query(q)
        
        if result_df is not None:
            print("✅ Result found:")
            print(result_df.to_string())
        
        print("-" * 70)