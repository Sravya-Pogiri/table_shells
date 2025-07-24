import os
import pandas as pd
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, Document

# --- Docling Imports ---
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    # Add this import for TableFormerMode if it's available in your Docling version
    from docling.datamodel.pipeline_options import TableFormerMode # You might need to check your Docling version for this.
except ImportError as e:
    print(f"Error importing Docling components: {e}")
    print("Please install Docling using one of these methods:")
    print("1. Basic installation: pip install docling")
    print("2. With OCR support: pip install 'docling[ocr]'")
    print("3. Full installation: pip install 'docling[all]'")
    print("\nFor more information, visit: https://docling-project.github.io/docling/installation/")
    exit()

# --- LlamaIndex Import ---
try:
    from llama_index.core import Document
except ImportError:
    try:
        from llama_index.schema import Document
    except ImportError as e:
        print(f"Error importing LlamaIndex Document: {e}")
        print("Please install LlamaIndex: pip install llama-index")
        exit()


def combine_tables(tables, similarity_threshold=0.7):
    """
    Combine multiple tables that should be one table.
    Uses column similarity to determine which tables to combine.
    
    Args:
        tables: List of table objects from Docling
        similarity_threshold: Threshold for considering tables similar enough to combine
    
    Returns:
        List of combined DataFrames
    """
    if not tables:
        return []
    
    # Convert all tables to DataFrames first
    table_dfs = []
    for i, table in enumerate(tables):
        try:
            df = table.export_to_dataframe()
            if not df.empty:
                table_dfs.append({
                    'index': i,
                    'dataframe': df,
                    'original_table': table
                })
        except Exception as e:
            print(f"Error converting table {i} to DataFrame: {e}")
            continue
    
    if not table_dfs:
        return []
    
    def calculate_column_similarity(df1, df2):
        """Calculate similarity between two DataFrames based on column names and structure"""
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        if len(cols1) == 0 and len(cols2) == 0:
            return 1.0
        
        intersection = cols1.intersection(cols2)
        union = cols1.union(cols2)
        
        return len(intersection) / len(union) if union else 0.0
    
    # Group similar tables
    combined_groups = []
    used_indices = set()
    
    for i, table_info in enumerate(table_dfs):
        if i in used_indices:
            continue
            
        current_group = [table_info]
        used_indices.add(i)
        
        # Find similar tables to combine
        for j, other_table_info in enumerate(table_dfs):
            if j in used_indices or i == j:
                continue
                
            similarity = calculate_column_similarity(
                table_info['dataframe'], 
                other_table_info['dataframe']
            )
            
            if similarity >= similarity_threshold:
                current_group.append(other_table_info)
                used_indices.add(j)
        
        combined_groups.append(current_group)
    
    # Combine tables in each group
    final_combined_tables = []
    for group_idx, group in enumerate(combined_groups):
        if len(group) == 1:
            # Single table, no combination needed
            final_combined_tables.append({
                'dataframe': group[0]['dataframe'],
                'original_indices': [group[0]['index']],
                'combined': False
            })
        else:
            # Multiple tables to combine
            try:
                # Sort by original index to maintain order
                group.sort(key=lambda x: x['index'])
                
                # Combine DataFrames
                combined_df = pd.concat(
                    [item['dataframe'] for item in group], 
                    ignore_index=True, 
                    sort=False
                )
                
                # Remove duplicate rows if any
                combined_df = combined_df.drop_duplicates().reset_index(drop=True)
                
                final_combined_tables.append({
                    'dataframe': combined_df,
                    'original_indices': [item['index'] for item in group],
                    'combined': True
                })
                
                print(f"Combined tables {[item['index'] for item in group]} into one table")
                print(f"Combined table shape: {combined_df.shape}")
                
            except Exception as e:
                print(f"Error combining tables in group {group_idx}: {e}")
                # Fall back to individual tables
                for item in group:
                    final_combined_tables.append({
                        'dataframe': item['dataframe'],
                        'original_indices': [item['index']],
                        'combined': False
                    })
    
    return final_combined_tables


def read_document_with_docling_combined_tables(file_path, combine_tables_flag=True, similarity_threshold=0.7):
    """
    Reads text and tables from a PDF or DOCX file using Docling,
    with option to combine similar tables into one.
    Returns a list of LlamaIndex Document objects.
    """
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        # Optionally, try ACCURATE mode for better table structure recognition
        # pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        print(f"Processing document with Docling: {file_path}")
        conv_result = doc_converter.convert(file_path)
        doc = conv_result.document

        llama_documents = []
        print(f"Found {len(doc.tables)} tables in the document")

        # Combine tables if requested
        if combine_tables_flag and doc.tables:
            combined_tables = combine_tables(doc.tables, similarity_threshold)
            print(f"After combination: {len(combined_tables)} table(s)")
        else:
            # Convert to the same format for consistency
            combined_tables = []
            for i, table in enumerate(doc.tables):
                try:
                    df = table.export_to_dataframe()
                    combined_tables.append({
                        'dataframe': df,
                        'original_indices': [i],
                        'combined': False
                    })
                except Exception as e:
                    print(f"Error processing table {i}: {e}")

        # Process combined tables
        for table_idx, table_info in enumerate(combined_tables):
            try:
                table_df = table_info['dataframe']
                original_indices = table_info['original_indices']
                is_combined = table_info['combined']

                table_metadata = {
                    "element_type": "table",
                    "table_index": table_idx,
                    "original_table_indices": original_indices,
                    "is_combined": is_combined,
                    "page_number": "N/A"
                }

                # Try to get page number from first original table
                if original_indices and len(doc.tables) > original_indices[0]:
                    original_table = doc.tables[original_indices[0]]
                    if hasattr(original_table, 'prov') and original_table.prov:
                        for prov_item in original_table.prov:
                            if hasattr(prov_item, 'page'):
                                table_metadata["page_number"] = prov_item.page
                                break

                table_text_tabular = table_df.to_string(index=True)
                table_text_csv = table_df.to_csv(index=True)
                table_markdown = table_df.to_markdown(index=True)

                # Create title based on whether it's combined or not
                if is_combined:
                    title_prefix = f"Combined Table {table_idx + 1} (from original tables {original_indices})"
                else:
                    title_prefix = f"Table {table_idx + 1}"

                llama_documents.append(Document(
                    text=f"{title_prefix}:\n{table_text_tabular}",
                    metadata={**table_metadata, "format": "tabular"}
                ))

                llama_documents.append(Document(
                    text=f"{title_prefix} (CSV format):\n{table_text_csv}",
                    metadata={**table_metadata, "format": "csv"}
                ))

                llama_documents.append(Document(
                    text=f"{title_prefix} (Markdown):\n{table_markdown}",
                    metadata={**table_metadata, "format": "markdown"}
                ))

                print(f"Successfully processed {title_prefix} with {len(table_df)} rows and {len(table_df.columns)} columns")
                print(f"Table preview:\n{table_df.head()}")

            except Exception as e:
                print(f"Error processing combined table {table_idx}: {e}")

        # Process other document elements (non-table content)
        for element in doc.body:
            element_type = element.__class__.__name__.lower()
            if 'table' in element_type:
                continue

            if hasattr(element, 'text') and element.text and element.text.strip():
                page_number = getattr(element, 'page', 'N/A')
                if hasattr(element, 'prov') and element.prov:
                    for prov_item in element.prov:
                        if hasattr(prov_item, 'page'):
                            page_number = prov_item.page
                            break

                metadata = {
                    "element_type": element_type,
                    "page_number": page_number
                }

                if hasattr(element, 'label'):
                    metadata["label"] = element.label
                if hasattr(element, 'level') and element_type in ['heading', 'title']:
                    metadata["heading_level"] = element.level

                llama_documents.append(Document(
                    text=element.text,
                    metadata=metadata
                ))

        # Fallback if no documents were created
        if not llama_documents:
            full_text = doc.export_to_text()
            if full_text and full_text.strip():
                llama_documents.append(Document(
                    text=full_text,
                    metadata={
                        "element_type": "full_document",
                        "page_number": "all",
                        "format": "fallback"
                    }
                ))

        print(f"Created {len(llama_documents)} documents from Docling parsing")
        return llama_documents

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting content from {file_path} with Docling: {e}")
        print(f"Error type: {type(e).__name__}")
        return []


def export_combined_tables_directly(file_path, output_dir="./table_exports", combine_tables_flag=True, similarity_threshold=0.7):
    """
    First combines tables based on similarity, then exports the combined results to storage folder.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        print(f"Processing document: {file_path}")
        conv_result = doc_converter.convert(file_path)
        doc = conv_result.document

        doc_filename = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Found {len(doc.tables)} original tables in document")

        # STEP 1: Always combine tables first if there are any tables
        combined_tables = []
        if doc.tables:
            if combine_tables_flag:
                print("Combining similar tables...")
                combined_tables = combine_tables(doc.tables, similarity_threshold)
                print(f"After combination: {len(combined_tables)} table(s)")
            else:
                print("Keeping tables separate (combine_tables_flag=False)")
                # Still process them through the same format for consistency
                for i, table in enumerate(doc.tables):
                    try:
                        df = table.export_to_dataframe()
                        combined_tables.append({
                            'dataframe': df,
                            'original_indices': [i],
                            'combined': False
                        })
                    except Exception as e:
                        print(f"Error processing table {i}: {e}")

        if not combined_tables:
            print("No tables found or all tables failed to process")
            return []

        # STEP 2: Export the combined tables to storage folder
        exported_tables = []
        print(f"\nExporting {len(combined_tables)} table(s) to {output_dir}...")

        for table_idx, table_info in enumerate(combined_tables):
            try:
                table_df = table_info['dataframe']
                original_indices = table_info['original_indices']
                is_combined = table_info['combined']

                if table_df.empty:
                    print(f"Skipping empty table {table_idx + 1}")
                    continue

                # Handle duplicate column names for JSON export
                cols = pd.Series(table_df.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(len(cols[cols == dup]))]
                table_df.columns = cols

                # Create descriptive filename
                if is_combined:
                    suffix = f"combined_table_{table_idx + 1}_from_originals_{'-'.join(map(str, original_indices))}"
                    table_description = f"Combined Table {table_idx + 1} (merged from original tables: {original_indices})"
                else:
                    suffix = f"table_{table_idx + 1}"
                    table_description = f"Table {table_idx + 1}"

                # Export in multiple formats
                base_filename = f"{doc_filename}_{suffix}"

                # CSV Export
                csv_filename = os.path.join(output_dir, f"{base_filename}.csv")
                table_df.to_csv(csv_filename, index=True)

                # JSON Export
                json_filename = os.path.join(output_dir, f"{base_filename}.json")
                table_df.to_json(json_filename, orient='records', indent=2)

                # Markdown Export
                md_filename = os.path.join(output_dir, f"{base_filename}.md")
                with open(md_filename, 'w', encoding='utf-8') as f:
                    f.write(f"# {table_description}\n\n")
                    f.write(f"**Shape:** {table_df.shape[0]} rows × {table_df.shape[1]} columns\n\n")
                    if is_combined:
                        f.write(f"**Note:** This table was created by combining original tables: {original_indices}\n\n")
                    f.write(table_df.to_markdown(index=True))

                # HTML Export
                html_filename = os.path.join(output_dir, f"{base_filename}.html")
                html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{table_description}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .info {{ background-color: #e7f3ff; padding: 10px; margin-bottom: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{table_description}</h1>
    <div class="info">
        <strong>Shape:</strong> {table_df.shape[0]} rows × {table_df.shape[1]} columns<br>
        {'<strong>Combined from original tables:</strong> ' + str(original_indices) + '<br>' if is_combined else ''}
    </div>
    {table_df.to_html(index=True, escape=False)}
</body>
</html>
                """
                with open(html_filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                # Excel Export (bonus format)
                excel_filename = os.path.join(output_dir, f"{base_filename}.xlsx")
                try:
                    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                        table_df.to_excel(writer, sheet_name='Data', index=True)
                        
                        # Add metadata sheet
                        metadata_df = pd.DataFrame({
                            'Property': ['Description', 'Shape', 'Original Tables', 'Is Combined', 'Export Date'],
                            'Value': [
                                table_description,
                                f"{table_df.shape[0]} rows × {table_df.shape[1]} columns",
                                str(original_indices),
                                str(is_combined),
                                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                            ]
                        })
                        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                except Exception as e:
                    print(f"Could not export Excel file: {e}")
                    excel_filename = None

                exported_tables.append({
                    'table_index': table_idx,
                    'dataframe': table_df,
                    'original_indices': original_indices,
                    'is_combined': is_combined,
                    'description': table_description,
                    'files': {
                        'csv': csv_filename,
                        'json': json_filename,
                        'markdown': md_filename,
                        'html': html_filename,
                        'excel': excel_filename
                    }
                })

                print(f"✓ {table_description} exported successfully")
                if is_combined:
                    print(f"  └─ Merged from original tables: {original_indices}")
                print(f"  └─ Shape: {table_df.shape[0]} rows × {table_df.shape[1]} columns")
                print(f"  └─ Files: CSV, JSON, HTML, Markdown" + (", Excel" if excel_filename else ""))
                print(f"  └─ Preview:\n{table_df.head(3).to_string()}")
                print("-" * 70)

            except Exception as e:
                print(f"Error exporting table {table_idx}: {e}")

        print(f"\n🎉 Successfully exported {len(exported_tables)} combined table(s) to: {output_dir}")
        
        # Create summary file
        summary_filename = os.path.join(output_dir, f"{doc_filename}_export_summary.txt")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(f"Table Export Summary for: {doc_filename}\n")
            f.write(f"Export Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original Tables Found: {len(doc.tables)}\n")
            f.write(f"Final Tables After Combination: {len(exported_tables)}\n\n")
            
            for table in exported_tables:
                f.write(f"{table['description']}\n")
                f.write(f"  Shape: {table['dataframe'].shape[0]} rows × {table['dataframe'].shape[1]} columns\n")
                f.write(f"  Combined: {'Yes' if table['is_combined'] else 'No'}\n")
                if table['is_combined']:
                    f.write(f"  Source Tables: {table['original_indices']}\n")
                f.write(f"  Files: {', '.join([k.upper() for k, v in table['files'].items() if v])}\n\n")
        
        print(f"📋 Export summary saved to: {summary_filename}")
        return exported_tables

    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return []


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="mistral", request_timeout=1000.0, format="json")

# Usage example:
if __name__ == "__main__":
    file_path = "/Users/Sravya/Desktop/AI_model_table_shells/mark - can you convert this into a markdown.csv"

    print("\n--- Exporting combined tables directly ---")
    # Set combine_tables_flag=True to combine similar tables, False to keep them separate
    # Adjust similarity_threshold (0.0 to 1.0) - higher values require more similarity
    tables = export_combined_tables_directly(
        file_path, 
        combine_tables_flag=True, 
        similarity_threshold=0.7
    )

    print("\n--- Reading document with combined tables for LlamaIndex ---")
    llama_documents = read_document_with_docling_combined_tables(
        file_path, 
        combine_tables_flag=True, 
        similarity_threshold=0.7
    )
    if llama_documents:
        print(f"Successfully created {len(llama_documents)} LlamaIndex documents.")
    else:
        print("No LlamaIndex documents were created.")