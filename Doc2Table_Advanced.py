# --- Core Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import openai
import re
import io
import os
import tempfile
import json
import datetime


# --- DOCX Processing ---
try:
   from docx import Document as DocxDocument
   from docx.table import _Cell
except ImportError:
   st.error("Please run: pip install python-docx")


st.set_page_config(layout="wide", page_title="TableGen AI")


class SAPEmbedsWeb:
   def __init__(self):
       # LAZY IMPORT: Only load sklearn when this class is actually instantiated
       try:
           from sklearn.feature_extraction.text import TfidfVectorizer
       except ImportError:
           st.error("Please install scikit-learn: pip install scikit-learn")
           return


       self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
       self.chunks = []
       self.vectors = None


   def process_docx(self, file_path: str, chunk_length: int = 400, chunk_overlap: int = 80):
       # LAZY IMPORT
       from sklearn.metrics.pairwise import cosine_similarity
     
       try:
           doc = DocxDocument(file_path)
           full_text = ""
        
           # Extract text from paragraphs
           for para in doc.paragraphs:
               full_text += para.text + "\n"
        
           # Extract text from tables
           for table in doc.tables:
               for row in table.rows:
                   for cell in row.cells:
                       full_text += cell.text + " "
                   full_text += "\n"


           chunks = []
           start = 0
           while start < len(full_text):
               end = min(start + chunk_length, len(full_text))
               chunks.append(full_text[start:end])
               start += chunk_length - chunk_overlap


           self.chunks = chunks
           if chunks:
               self.vectors = self.vectorizer.fit_transform(chunks)
           return chunks
       except Exception as e:
           st.error(f"Error processing DOCX: {e}")
           return []


   def retrieve(self, query, top=1):
       from sklearn.metrics.pairwise import cosine_similarity
     
       if not self.chunks or self.vectors is None:
           return []


       query_vector = self.vectorizer.transform([query])
       similarities = cosine_similarity(query_vector, self.vectors)[0]
       top_indices = np.argsort(similarities)[-top:][::-1]
       return [self.chunks[i] for i in top_indices]


   def analyzeLLM(self, query):
       best_chunks = self.retrieve(query, top=1)
       if not best_chunks:
           return ["Could not find a relevant chunk in the document for your query."]


       results = []
       for chunk in best_chunks:
           prompt = f"""Answer the question using ONLY the given text chunk. Your primary goal is to extract and list the specific variables or items requested. Do not use outside information.


           Question: {query}


           Text Chunk:
           ---
           {chunk}
           ---


           Answer (list the variables):"""


           try:
               client = openai.Client(
                   base_url="https://api.groq.com/openai/v1",
                   api_key=st.secrets["GROQ_API_KEY"]
               )
               response = client.chat.completions.create(
                   model="llama-3.3-70b-versatile",
                   messages=[{'role': 'user', 'content': prompt}]
               )
               clean_response = self.clean_response(response.choices[0].message.content)
               results.append(clean_response)
           except Exception as e:
               results.append(f"Error communicating with LLM: {str(e)}")


       return results


   def clean_response(self, response_text):
       lines = response_text.split('\n')
       variables = []
       for line in lines:
           line = line.strip()
           if re.match(r'^\d+\.\s*', line) or re.match(r'^[\*\-]\s*', line):
               clean_var = re.sub(r'^\d+\.\s*|^[\*\-]\s*', '', line).strip()
               if clean_var and len(clean_var) > 1:
                   variables.append(clean_var)
       return variables if variables else [response_text]


def run_variable_extractor_app():
   st.header("1. TableGen AI: Variable Extractor")
   st.markdown("Upload a document, then ask questions to find and extract key variables.")


   uploaded_file = st.file_uploader("Upload a DOCX Document", type='docx', key='var_extractor_uploader')


   if 'sap_model' not in st.session_state:
       st.session_state.sap_model = None


   if uploaded_file:
       if st.session_state.sap_model is None:
            st.session_state.sap_model = SAPEmbedsWeb()


       if st.session_state.get('last_uploaded_file_name') != uploaded_file.name:
           with st.spinner(f"Processing '{uploaded_file.name}'..."):
               with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                   tmp_file.write(uploaded_file.read())
                   tmp_path = tmp_file.name
             
               st.session_state.chunks = st.session_state.sap_model.process_docx(tmp_path)
               os.unlink(tmp_path)
         
           st.success(f"Loaded {len(st.session_state.chunks)} chunks from '{uploaded_file.name}'")
           st.session_state.last_uploaded_file_name = uploaded_file.name
           st.session_state.chat_history = []
           st.rerun()


   if 'chunks' in st.session_state and st.session_state.chunks:
       st.subheader("❓ Ask a Question")
       sample_questions = [
           "What are the variables in the demographics section?",
           "What are the prior cancer therapies mentioned?",
           "What are the medical history variables?",
           "What are the stratification factors?",
       ]
       cols = st.columns(len(sample_questions))
       for i, q in enumerate(sample_questions):
           if cols[i].button(q, key=f"sample_{i}"):
               st.session_state.current_question = q
               st.session_state.query_submitted = True
    
       question = st.text_input("Enter your question:", value=st.session_state.get('current_question', ''), key="question_input")
    
       if st.button("Ask Question", type="primary") or st.session_state.get('query_submitted'):
           if question:
               st.session_state.query_submitted = False
               with st.spinner("Analyzing document..."):
                   results = st.session_state.sap_model.analyzeLLM(question)
               st.session_state.chat_history.insert(0, (question, results))
               st.session_state.current_question = ""


       if st.session_state.get('chat_history'):
           st.subheader("Latest Results")
           latest_q, latest_a = st.session_state.chat_history[0]
           with st.container(border=True):
               st.write(f"**Your Question:** {latest_q}")
               st.divider()
               st.write("**Extracted Variables:**")
               for result_group in latest_a:
                   if isinstance(result_group, list):
                       for var in result_group:
                           st.markdown(f"- {var}")
                   else:
                       st.write(result_group)




# ##############################################################################
# --- APP 2: TABLE SHELL GENERATOR WITH FULL RAG (INCLUDING FORMAT DETECTION) ---
# ##############################################################################


class TableRAGSystem:
   """RAG system for table structure extraction WITH format detection"""
   def __init__(self):
       try:
           from sklearn.feature_extraction.text import TfidfVectorizer
           from sklearn.metrics.pairwise import cosine_similarity
       except ImportError:
           st.error("Please install scikit-learn: pip install scikit-learn")
           return
      
       self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
       self.table_chunks = []  # Store table content chunks
       self.table_metadata = []  # Store metadata (title, position)
       self.vectors = None
  
   def extract_tables_from_docx(self, file_path: str):
       """Extract tables and their context from DOCX"""
       doc = DocxDocument(file_path)
      
       current_table_title = "Unknown Table"
       table_count = 0
      
       for element in doc.element.body:
           if element.tag.endswith('p'):
               para = next((p for p in doc.paragraphs if p._element == element), None)
               if para and para.text.strip():
                   text = para.text.strip()
                   # Normalize title
                   clean_text = re.sub(r'\s+', ' ', text).strip()
                   clean_text = re.sub(r'^[.\s]+', '', clean_text).strip()
                  
                   if re.match(r'^Table\s', clean_text, re.IGNORECASE) or len(clean_text) < 100:
                       current_table_title = clean_text
          
           elif element.tag.endswith('tbl'):
               table = next((t for t in doc.tables if t._element == element), None)
               if table and len(table.rows) > 1:
                   table_count += 1
                  
                   # Extract table content as text
                   table_text = self._extract_table_text(table)
                  
                   # Store chunk and metadata
                   self.table_chunks.append(table_text)
                   self.table_metadata.append({
                       'title': current_table_title,
                       'table_number': table_count,
                       'table_object': table
                   })
      
       # Create embeddings
       if self.table_chunks:
           self.vectors = self.vectorizer.fit_transform(self.table_chunks)
      
       return len(self.table_chunks)
  
   def _extract_table_text(self, table) -> str:
       """Convert table to text representation"""
       table_rows = []
       for row_idx, row in enumerate(table.rows):
           cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
           if cells:
               row_text = " | ".join(cells)
               table_rows.append(f"Row {row_idx}: {row_text}")
       return "\n".join(table_rows)
  
   def retrieve_relevant_tables(self, query: str, top_k: int = 10):
       """Retrieve most relevant tables using semantic search"""
       from sklearn.metrics.pairwise import cosine_similarity
      
       if not self.table_chunks or self.vectors is None:
           return []
      
       query_vector = self.vectorizer.transform([query])
       similarities = cosine_similarity(query_vector, self.vectors)[0]
       top_indices = np.argsort(similarities)[-top_k:][::-1]
      
       results = []
       for idx in top_indices:
           # Lower threshold to 0.05 to catch more relevant tables
           if similarities[idx] > 0.05:
               results.append({
                   'chunk': self.table_chunks[idx],
                   'metadata': self.table_metadata[idx],
                   'similarity': similarities[idx]
               })
      
       return results
  
   def extract_structure_with_llm(self, table_text: str, table_title: str) -> list:
       """Use LLM to extract variable structure AND appropriate formats from retrieved table"""
       prompt = f"""You are an expert clinical data manager. Analyze this table content and structure it into Variables with their Row Labels AND appropriate placeholder formats.


       CRITICAL INSTRUCTIONS:
       1. Identify the **Parent Variable** (main category like "Age", "Sex", "ECOG Performance Status", "Prior Radiation Therapy").
       2. For each parent variable, list its **Row Labels** with the CORRECT format placeholder.
       3. For categorical variables (e.g., Sex), the rows are the categories (e.g., "Male", "Female").
       4. **DO NOT** create separate variables for hierarchical breakdowns. For example:
          - "Any Prior Systemic Cancer Therapy" with sub-items "Class 1", "Subclass 1", "Drug 1" should be ONE variable
          - All the drugs/classes should be listed as rows under the parent variable
       5. **SPECIAL CASE - Nested Hierarchies (Medical History, Adverse Events)**:
          - If the table has "System Organ Class" AND "Preferred Term", combine them into ONE variable
          - Structure: "System Organ Class" as the parent with its categories as rows
          - Under each System Organ Class category, list "Preferred Term 1", "Preferred Term 2", etc. as sub-rows
          - Example: "System Organ Class" → "Subjects with any Report" → "    Preferred Term 1" → "    Preferred Term 2"
       6. Look for patterns like:
          - Bold/indented rows indicate hierarchy (keep them together)
          - Statistical summaries (n, Mean, SD, Median, Min/Max) belong to their parent
          - Sub-categories (like "Neo-Adjuvant", "Metastatic") belong to their parent
       7. Ignore column headers (Treatment A, Treatment B, Total).


       FORMAT PLACEHOLDERS (CRITICAL):
       - For count/n: use "xx"
       - For continuous statistics (Mean, Median, SD, etc.): use "xxx.xx"
       - For Min/Max ranges: use "xxx, xxx"
       - For categorical counts with percentages: use "xx (xx.x)"
       - For percentiles/quartiles: use "xxx.xx"


       Table Title: {table_title}
       Table Content:
       {table_text[:5000]} (truncated)


       Respond with a valid JSON list of objects. Do not add markdown formatting or conversational text.
       Format:
       [
         {{
           "variable": "Height (cm)",
           "rows": [
             {{"label": "n", "format": "xx"}},
             {{"label": "Mean", "format": "xxx.xx"}},
             {{"label": "Standard Deviation", "format": "xxx.xx"}},
             {{"label": "Median", "format": "xxx.xx"}},
             {{"label": "Min, Max", "format": "xxx, xxx"}}
           ]
         }},
         {{
           "variable": "Sex",
           "rows": [
             {{"label": "Male", "format": "xx (xx.x)"}},
             {{"label": "Female", "format": "xx (xx.x)"}}
           ]
         }},
         {{
           "variable": "System Organ Class",
           "rows": [
             {{"label": "Subjects with any Report", "format": "xx (xx.x)"}},
             {{"label": "System Organ Class 1", "format": "xx (xx.x)"}},
             {{"label": "    Preferred Term 1", "format": "xx (xx.x)"}},
             {{"label": "    Preferred Term 2", "format": "xx (xx.x)"}},
             {{"label": "System Organ Class 2", "format": "xx (xx.x)"}},
             {{"label": "    Preferred Term 1", "format": "xx (xx.x)"}}
           ]
         }}
       ]


       JSON:"""


       try:
           client = openai.Client(
               base_url="https://api.groq.com/openai/v1",
               api_key=st.secrets["GROQ_API_KEY"]
           )
          
           response = client.chat.completions.create(
               model="llama-3.3-70b-versatile",
               messages=[{'role': 'user', 'content': prompt}],
               temperature=0.0,
               max_tokens=2000
           )
          
           result_text = response.choices[0].message.content.strip()
          
           # Extract JSON
           match = re.search(r'\[.*\]', result_text, re.DOTALL)
           if match:
               json_str = match.group(0)
               extracted_data = json.loads(json_str)
               return extracted_data
           else:
               clean_text = re.sub(r'```json|```', '', result_text).strip()
               if clean_text:
                   return json.loads(clean_text)
          
           return []
          
       except json.JSONDecodeError:
           print(f"JSON Parse Error. LLM returned: {result_text}")
           return []
       except Exception as e:
           print(f"General Error: {e}")
           return []


def consolidate_hierarchical_variables(data: dict) -> dict:
   """Merge variables that are clearly hierarchical sub-items into their parent"""
   consolidated = {}
  
   for table_title, items in data.items():
       # SPECIAL CASE: If we have both "System Organ Class" and "Preferred Term", merge them
       has_soc = any('system organ class' in item.get('variable', '').lower() for item in items)
       has_pt = any('preferred term' in item.get('variable', '').lower() for item in items)
      
       if has_soc and has_pt:
           # Merge System Organ Class and Preferred Term into one variable
           soc_items = [item for item in items if 'system organ class' in item.get('variable', '').lower()]
           pt_items = [item for item in items if 'preferred term' in item.get('variable', '').lower()]
          
           if soc_items:
               # Take the first SOC as the base
               merged_soc = soc_items[0].copy()
              
               # Add all PT rows to SOC rows with proper indentation
               for pt_item in pt_items:
                   pt_rows = pt_item.get('rows', [])
                   for row in pt_rows:
                       if isinstance(row, dict):
                           # Add indentation if not already present
                           label = row.get('label', '')
                           if not label.startswith('    '):
                               row['label'] = f"    {label}"
                       merged_soc['rows'].append(row)
              
               # Keep only non-SOC, non-PT items plus the merged one
               other_items = [item for item in items
                             if 'system organ class' not in item.get('variable', '').lower()
                             and 'preferred term' not in item.get('variable', '').lower()]
              
               consolidated[table_title] = [merged_soc] + other_items
               continue
      
       # STANDARD PROCESSING: Keywords-based merging
       parent_keywords = [
           "any prior", "lines of", "number of", "intent of",
           "best response", "reason for", "type of", "location of",
           "history of", "subjects with"
       ]
      
       # Keywords that indicate a row is a sub-item (should be merged)
       subitem_keywords = ["class", "subclass", "drug", "agent", "regimen"]
      
       merged_items = []
       current_parent = None
      
       for item in items:
           var_name = item.get('variable', '').lower()
          
           # Check if this is a parent variable
           is_parent = any(keyword in var_name for keyword in parent_keywords)
          
           # Check if this is a subitem that should be merged
           is_subitem = any(keyword in var_name for keyword in subitem_keywords)
          
           if is_parent:
               # Start new parent
               current_parent = item
               merged_items.append(item)
           elif is_subitem and current_parent:
               # Merge into current parent
               current_parent['rows'].extend(item.get('rows', []))
           else:
               # Standalone variable
               merged_items.append(item)
               current_parent = None
      
       consolidated[table_title] = merged_items
  
   return consolidated


def parse_docx_with_rag(file_path: str) -> dict:
   """Parse DOCX using TRUE RAG approach - retrieve relevant tables for each unique title"""
   # Initialize RAG system
   rag_system = TableRAGSystem()
  
   # Extract and index all tables
   with st.spinner("📚 Indexing tables for RAG retrieval..."):
       num_tables = rag_system.extract_tables_from_docx(file_path)
       st.info(f"Indexed {num_tables} tables")
  
   # Get all unique table titles from metadata
   unique_titles = list(set(meta['title'] for meta in rag_system.table_metadata))
  
   # Group tables by title using RAG retrieval
   hierarchical_data = {}
  
   with st.spinner("🔍 Using TRUE RAG to retrieve and extract structure + formats..."):
       for title in unique_titles:
           if title not in hierarchical_data:
               hierarchical_data[title] = []
          
           # TRUE RAG: Query the vector store for tables matching this title
           query = f"{title} demographic baseline characteristics variables"
           relevant_results = rag_system.retrieve_relevant_tables(query, top_k=15)
          
           # Track if we found any results via RAG
           found_via_rag = False
          
           # Process each retrieved table that matches the title
           for result in relevant_results:
               result_title = result['metadata']['title']
              
               # Only process if it's the same table (normalized title match)
               if result_title == title:
                   found_via_rag = True
                   # Extract structure using LLM (now includes format)
                   structure = rag_system.extract_structure_with_llm(
                       result['chunk'],
                       result_title
                   )
                  
                   if structure:
                       hierarchical_data[title].extend(structure)
          
           # FALLBACK: If RAG didn't find anything, search metadata directly
           if not found_via_rag:
               for idx, meta in enumerate(rag_system.table_metadata):
                   if meta['title'] == title:
                       chunk = rag_system.table_chunks[idx]
                       structure = rag_system.extract_structure_with_llm(chunk, title)
                       if structure:
                           hierarchical_data[title].extend(structure)
  
   # POST-PROCESSING: Consolidate hierarchical variables
   hierarchical_data = consolidate_hierarchical_variables(hierarchical_data)
  
   return hierarchical_data


def process_docx_fast(file_content, file_name):
   with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
       tmp.write(file_content)
       tmp_path = tmp.name


   try:
       st.info(f"📄 Reading '{file_name}' using RAG + LLM...")
       # Use RAG-based parsing
       hierarchical_data = parse_docx_with_rag(tmp_path)
      
       st.success(f"✅ Processed {len(hierarchical_data)} unique table titles")
      
       with st.expander("📋 RAG-Identified Categories", expanded=True):
           for table_title, items in hierarchical_data.items():
               st.write(f"**{table_title}**")
              
               # Quick stats on merged vars
               unique_vars = set(i.get('variable', '') for i in items)
               st.write(f"  • {len(unique_vars)} unique variables detected")
               st.divider()


       return hierarchical_data


   except Exception as e:
       st.error(f"❌ Error processing document: {e}")
       return {}
   finally:
       if os.path.exists(tmp_path):
           os.remove(tmp_path)


def run_table_shell_app():
   st.header("2. 🤖 Full RAG + LLM Table Shell Generator")
   st.markdown("Upload a **DOCX** file. Uses RAG retrieval + LLM to intelligently extract table structures **and format placeholders**.")


   uploaded_table_file = st.file_uploader("**Upload Clinical DOCX**", type='docx', key='table_uploader_docx')


   if 'doc_data' not in st.session_state:
       st.session_state.doc_data = {}


   if uploaded_table_file:
       file_key = f"processed_{uploaded_table_file.name}"
       if st.session_state.get('last_processed_file') != file_key:
           st.session_state.generated_tables = {}
           st.session_state.table_order = []
          
           # Process the file with RAG
           data = process_docx_fast(uploaded_table_file.getvalue(), uploaded_table_file.name)
          
           if data:
               st.session_state.doc_data = data
               st.session_state.last_processed_file = file_key
               st.success("✅ Document processed with RAG!")
               st.rerun()


   if not st.session_state.doc_data:
       st.info("📤 Please upload a DOCX file to start")
       return
 
   doc_data = st.session_state.doc_data
 
   if 'generated_tables' not in st.session_state:
       st.session_state.generated_tables = {}
   if 'table_order' not in st.session_state:
       st.session_state.table_order = []
 
   # 1. Select Table
   table_options = list(doc_data.keys())
   if not table_options:
       st.warning("No tables found in the processed document.")
       return


   if 'selected_main_category' not in st.session_state or st.session_state.selected_main_category not in table_options:
       st.session_state.selected_main_category = table_options[0]


   selected_table_name = st.selectbox(
       "**Step 1: Select a Table**",
       options=table_options,
       index=table_options.index(st.session_state.selected_main_category)
   )
   st.session_state.selected_main_category = selected_table_name


   # 2. Get Variables for that table
   if selected_table_name:
       table_structure = doc_data[selected_table_name] # List of dicts
      
       # Merge logic: consolidate duplicate variables while preserving formats
       var_map = {}
       for item in table_structure:
           v_name = item.get('variable', '').strip()
           if not v_name: continue
          
           v_rows = item.get('rows', [])
          
           if v_name in var_map:
               existing_rows = var_map[v_name]
               for r in v_rows:
                   # Check if label already exists
                   r_label = r.get('label') if isinstance(r, dict) else r
                   if not any(existing.get('label') == r_label if isinstance(existing, dict) else existing == r_label for existing in existing_rows):
                       existing_rows.append(r)
           else:
               var_map[v_name] = v_rows[:]


       variable_names = list(var_map.keys())
      
       # Determine defaults
       default_selections = [v for v in variable_names if v in st.session_state.generated_tables]


       selected_variables = st.multiselect(
           f"**Step 2: Select Variables to Create Shells For**",
           options=variable_names,
           default=default_selections
       )
     
       if st.button("Generate Tables", type="primary", disabled=not selected_variables):
           with st.spinner("Creating table shells using RAG-determined formats..."):
               for var_name in selected_variables:
                   if var_name not in st.session_state.generated_tables:
                      
                       rows = var_map.get(var_name, [])
                      
                       # Build columns with variable name as header
                       col_param = [var_name]
                       empty_data = ['']
                      
                       # Process each row with its RAG-determined format
                       for r in rows:
                           if isinstance(r, dict):
                               # RAG-determined format
                               label = r.get('label', '')
                               format_str = r.get('format', 'xx (xx.x)')
                               col_param.append(f"    {label}")
                               empty_data.append(format_str)
                           else:
                               # Fallback if format not provided (shouldn't happen with full RAG)
                               col_param.append(f"    {r}")
                               empty_data.append('xx (xx.x)')
                      
                       df = pd.DataFrame({
                           'Characteristic': col_param,
                           'Treatment A (N=xxx)': empty_data,
                           'Treatment B (N=xxx)': empty_data,
                           'Total (N=xxx)': empty_data
                       })
                      
                       st.session_state.generated_tables[var_name] = df
                       if var_name not in st.session_state.table_order:
                           st.session_state.table_order.append(var_name)
              
               for var in list(st.session_state.generated_tables.keys()):
                   if var in variable_names and var not in selected_variables:
                       del st.session_state.generated_tables[var]
                       if var in st.session_state.table_order:
                           st.session_state.table_order.remove(var)
          
           st.rerun()


   # Display Logic
   if st.session_state.generated_tables:
       st.subheader(f"📊 Table Shells ({len(st.session_state.generated_tables)})")
      
       for i, category in enumerate(st.session_state.table_order[:]):
           if category not in st.session_state.generated_tables:
               continue


           df = st.session_state.generated_tables[category]
           with st.container(border=True):
               col1, col2 = st.columns([0.85, 0.15])
               with col1:
                   st.markdown(f"**{category}**")
               with col2:
                   subcol1, subcol2, subcol3 = st.columns(3)
                   with subcol1:
                       if st.button("⬆️", key=f"u{i}", disabled=i==0):
                           st.session_state.table_order.insert(i-1, st.session_state.table_order.pop(i))
                           st.rerun()
                   with subcol2:
                       if st.button("⬇️", key=f"d{i}", disabled=i==len(st.session_state.table_order)-1):
                           st.session_state.table_order.insert(i+1, st.session_state.table_order.pop(i))
                           st.rerun()
                   with subcol3:
                       if st.button("🗑️", key=f"del{i}"):
                           del st.session_state.generated_tables[category]
                           st.session_state.table_order.remove(category)
                           st.rerun()
              
               edited_df = st.data_editor(
                   df,
                   key=f"editor_{category}",
                   num_rows="dynamic",
                   use_container_width=True,
                   hide_index=True
               )
               if not df.equals(edited_df):
                   st.session_state.generated_tables[category] = edited_df


       st.divider()
       col1, col2 = st.columns(2)
       with col1:
           output = io.StringIO()
           for category in st.session_state.table_order:
               df_save = st.session_state.generated_tables[category]
               output.write(f'"{category}"\n')
               df_save.to_csv(output, index=False)
               output.write("\n")
           st.download_button(
               "⬇️ Download CSV",
               output.getvalue().encode('utf-8'),
               "tables.csv",
               "text/csv",
               use_container_width=True
           )
       with col2:
           if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
               st.session_state.generated_tables = {}
               st.session_state.table_order = []
               st.rerun()


# ##############################################################################
# --- MAIN UI ---
# ##############################################################################


def main():
   st.title("TableGen AI: Clinical Data Assistant (Full RAG-Powered)")
    with st.sidebar:
       st.header("Controls")
       if st.button("Clear History"):
           st.session_state.chat_history = []
           st.rerun()
    
       st.info("⚡ Full RAG Mode Active\nLLM determines both structure AND format placeholders")


   run_variable_extractor_app()
   st.divider()
   run_table_shell_app()


if __name__ == "__main__":
   main()
