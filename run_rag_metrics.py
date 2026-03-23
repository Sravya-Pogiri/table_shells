import datetime
import io
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


PDF_PATH = "SAP.pdf"
TABLE_CSV_PATH = "mark - can you convert this into a markdown.csv"
OUTPUT_PNG = "rag_metrics.png"
SUMMARY_PNG = "rag_metrics_summary.png"
ACCURACY_PNG = "rag_metrics_accuracy.png"
OUTPUT_CSV = "rag_metrics.csv"
NUM_VARIABLE_QUERIES = 40
NUM_TABLE_QUERIES = 40
SUCCESS_SIMILARITY_THRESHOLD = 0.15


def extract_pdf_text(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text
    return full_text


def chunk_text(text: str, chunk_length: int = 400, chunk_overlap: int = 80) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_length, len(text))
        chunks.append(text[start:end])
        start += chunk_length - chunk_overlap
    return chunks


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return set(tokens)


def proxy_accuracy(query: str, chunk: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    chunk_tokens = tokenize(chunk)
    return len(query_tokens & chunk_tokens) / len(query_tokens)


def make_timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def extract_keywords(text: str, limit: int = 120) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    if not tokens:
        return []
    freq = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _count in ranked[:limit]]


def generate_variable_queries(text: str, target_count: int) -> list[str]:
    keywords = extract_keywords(text)
    if not keywords:
        return [
            "What are the demographic variables?",
            "What is the study design?",
            "What are the inclusion criteria?",
            "What are the safety endpoints?",
            "What are the stratification factors?",
        ]

    templates = [
        "What are the variables related to {topic}?",
        "Summarize the {topic} section variables.",
        "List key {topic} endpoints.",
        "What does the document say about {topic}?",
        "Extract variables for {topic}.",
    ]

    queries = []
    for i in range(target_count):
        keyword = keywords[i % len(keywords)]
        template = templates[i % len(templates)]
        queries.append(template.format(topic=keyword))
    return queries


def run_variable_extractor_metrics(pdf_path: str) -> pd.DataFrame:
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    vectors = vectorizer.fit_transform(chunks) if chunks else None

    queries = generate_variable_queries(text, NUM_VARIABLE_QUERIES)

    rows = []
    for query in queries:
        start_time = time.perf_counter()
        retrieval_hit = False
        similarity_score = 0.0
        accuracy = 0.0

        if vectors is not None:
            query_vector = vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, vectors)[0]
            if similarities.size:
                top_index = int(np.argmax(similarities))
                top_chunk = chunks[top_index]
                retrieval_hit = True
                similarity_score = float(similarities[top_index])
                accuracy = proxy_accuracy(query, top_chunk)

        latency_ms = (time.perf_counter() - start_time) * 1000
        rows.append(
            {
                "timestamp": make_timestamp(),
                "model": "VariableExtractor",
                "retrieval_hit": int(retrieval_hit),
                "success": int(similarity_score >= SUCCESS_SIMILARITY_THRESHOLD),
                "similarity_score": similarity_score,
                "accuracy": accuracy * 100.0,
                "latency_ms": latency_ms,
            }
        )

    return pd.DataFrame(rows)


def parse_clinical_data_hierarchical(table_df: pd.DataFrame) -> dict:
    sanitized_columns = [f"col_{i}" for i in range(len(table_df.columns))]
    table_df.columns = sanitized_columns

    records = table_df.to_dict("records")

    structured_data = {}
    current_main_table = None
    current_sub_table_name = None
    current_sub_table_rows = []
    main_table_headers = []

    def save_previous_sub_table():
        if current_main_table and current_sub_table_name and current_sub_table_rows and main_table_headers:
            df = pd.DataFrame(current_sub_table_rows, columns=main_table_headers)
            structured_data.setdefault(current_main_table, {})[current_sub_table_name] = df

    for row in records:
        cols = [str(val) if pd.notna(val) else "" for val in row.values()]
        first_col, second_col = cols[0].strip(), cols[1].strip()
        other_cols_str = " ".join(cols[2:])

        if re.match(r"^Table [\d\.]+:.*", first_col, re.IGNORECASE):
            save_previous_sub_table()
            current_main_table = first_col
            current_sub_table_name, current_sub_table_rows, main_table_headers = None, [], []
            continue
        if not current_main_table:
            continue

        if "Treatment A" in other_cols_str or "Total (N=" in other_cols_str:
            main_table_headers = [second_col or "Characteristic"] + [c.strip() for c in cols[2:]]
            if first_col:
                save_previous_sub_table()
                current_sub_table_name = first_col
                current_sub_table_rows = []
            continue

        if first_col and first_col.lower() not in ["parameter", "characteristics"]:
            save_previous_sub_table()
            current_sub_table_name = first_col
            current_sub_table_rows = []
            if any(c.strip() for c in cols[2:]):
                current_sub_table_rows.append([first_col] + cols[2:])
        elif second_col and current_sub_table_name:
            current_sub_table_rows.append([second_col] + cols[2:])

    save_previous_sub_table()
    return {k: v for k, v in structured_data.items() if v}


def generate_table_queries(table_names: list[str], target_count: int) -> list[str]:
    if not table_names:
        return []
    templates = [
        "Show me the table shell for {name}",
        "Return the full table for {name}",
        "Get table shell: {name}",
        "Retrieve the table named {name}",
        "Provide the table shell for {name}",
    ]
    queries = []
    for i in range(target_count):
        name = table_names[i % len(table_names)]
        template = templates[i % len(templates)]
        queries.append(template.format(name=name))
    return queries


def run_table_generator_metrics(csv_path: str) -> pd.DataFrame:
    main_table_df = pd.read_csv(csv_path, header=None)
    hierarchical_tables = parse_clinical_data_hierarchical(main_table_df)
    parsed_tables = {
        sub_name: df for main_cat in hierarchical_tables.values() for sub_name, df in main_cat.items()
    }

    if not parsed_tables:
        return pd.DataFrame()

    Settings.llm = None
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    documents = [
        Document(text=f"Table: {name}\n{df.to_markdown(index=False)}", metadata={"table_name": name})
        for name, df in parsed_tables.items()
    ]

    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=1)

    table_names = list(parsed_tables.keys())
    queries = generate_table_queries(table_names, min(NUM_TABLE_QUERIES, len(table_names)))

    rows = []
    for query in queries:
        start_time = time.perf_counter()
        results = retriever.retrieve(query)
        latency_ms = (time.perf_counter() - start_time) * 1000

        retrieval_hit = bool(results)
        accuracy = 0.0
        success = 0
        if results:
            top = results[0]
            retrieved_name = top.metadata.get("table_name", "")
            accuracy = 100.0 if retrieved_name and retrieved_name in query else 0.0
            success = int(accuracy == 100.0)

        rows.append(
            {
                "timestamp": make_timestamp(),
                "model": "TableGenerator",
                "retrieval_hit": int(retrieval_hit),
                "success": success,
                "accuracy": accuracy,
                "latency_ms": latency_ms,
            }
        )

    return pd.DataFrame(rows)


def render_plots(df: pd.DataFrame, output_path: str):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    models = sorted(df["model"].unique())

    fig, axes = plt.subplots(len(models), 3, figsize=(12, 4 * len(models)), squeeze=False)

    for row_idx, model in enumerate(models):
        model_df = df[df["model"] == model].sort_values("timestamp")
        axes[row_idx, 0].plot(model_df["timestamp"], model_df["retrieval_hit"], marker="o")
        axes[row_idx, 0].set_title(f"{model} Retrieval Hit Rate")
        axes[row_idx, 0].set_ylim(-0.05, 1.05)

        axes[row_idx, 1].plot(model_df["timestamp"], model_df["accuracy"], marker="o", color="#4c78a8")
        axes[row_idx, 1].set_title(f"{model} Accuracy (Proxy)")
        axes[row_idx, 1].set_ylim(0, 100)

        axes[row_idx, 2].plot(model_df["timestamp"], model_df["latency_ms"], marker="o", color="#f58518")
        axes[row_idx, 2].set_title(f"{model} Efficiency (Latency ms)")

        for col in range(3):
            axes[row_idx, col].tick_params(axis="x", rotation=30)
            axes[row_idx, col].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def render_summary_plots(df: pd.DataFrame, output_path: str):
    summary = df.groupby("model").agg(
        hit_rate=("retrieval_hit", "mean"),
        success_rate=("success", "mean"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_median=("accuracy", "median"),
        latency_mean=("latency_ms", "mean"),
        latency_p95=("latency_ms", lambda series: np.percentile(series, 95)),
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    summary["hit_rate"].plot(kind="bar", ax=axes[0], color="#4c78a8")
    axes[0].set_title("Retrieval Hit Rate")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.3)

    summary["accuracy_mean"].plot(kind="bar", ax=axes[1], color="#72b7b2")
    axes[1].set_title("Accuracy Mean")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, axis="y", alpha=0.3)

    summary["success_rate"].plot(kind="bar", ax=axes[2], color="#b279a2")
    axes[2].set_title("Success Rate")
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(True, axis="y", alpha=0.3)

    summary["latency_mean"].plot(kind="bar", ax=axes[3], color="#f58518")
    axes[3].set_title("Latency Mean (ms)")
    axes[3].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def print_summary(df: pd.DataFrame):
    summary = df.groupby("model").agg(
        hit_rate=("retrieval_hit", "mean"),
        success_rate=("success", "mean"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_median=("accuracy", "median"),
        latency_mean=("latency_ms", "mean"),
        latency_p95=("latency_ms", lambda series: np.percentile(series, 95)),
        samples=("latency_ms", "count"),
    )
    print("\nSummary metrics (per model):")
    print(summary.to_string(float_format=lambda val: f"{val:.2f}"))


def render_accuracy_plots(df: pd.DataFrame, output_path: str):
    summary = df.groupby("model").agg(
        success_rate=("success", "mean"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_median=("accuracy", "median"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    summary["success_rate"].plot(kind="bar", ax=axes[0], color="#b279a2")
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.3)

    summary["accuracy_mean"].plot(kind="bar", ax=axes[1], color="#72b7b2")
    axes[1].set_title("Accuracy Mean")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, axis="y", alpha=0.3)

    summary["accuracy_median"].plot(kind="bar", ax=axes[2], color="#4c78a8")
    axes[2].set_title("Accuracy Median")
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")
    if not os.path.exists(TABLE_CSV_PATH):
        raise FileNotFoundError(f"Table CSV not found: {TABLE_CSV_PATH}")

    var_df = run_variable_extractor_metrics(PDF_PATH)
    table_df = run_table_generator_metrics(TABLE_CSV_PATH)

    metrics_df = pd.concat([var_df, table_df], ignore_index=True)
    if metrics_df.empty:
        raise RuntimeError("No metrics were generated.")

    metrics_df.to_csv(OUTPUT_CSV, index=False)
    render_plots(metrics_df, OUTPUT_PNG)
    render_summary_plots(metrics_df, SUMMARY_PNG)
    render_accuracy_plots(metrics_df, ACCURACY_PNG)
    print_summary(metrics_df)
    print(f"Saved metrics plot to {OUTPUT_PNG}")
    print(f"Saved summary plot to {SUMMARY_PNG}")
    print(f"Saved accuracy plot to {ACCURACY_PNG}")
    print(f"Saved metrics CSV to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
