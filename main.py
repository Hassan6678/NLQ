from llama_cpp import Llama
import pandas as pd
import duckdb
import os

# Path to your GGUF model
MODEL_PATH = "models/llama-3-sqlcoder-8b.Q6_K.gguf"

# Load model (use n_threads according to your CPU)
# llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6)
llm = Llama(
    model_path="models/llama-3-sqlcoder-8b.Q6_K.gguf",
    n_ctx=1024,  # Lower context size if RAM is an issue
    n_threads=6,
    n_gpu_layers=20,  # Safer for 8GB GPU
    verbose=True
)

# Load CSV into DuckDB
df = pd.read_csv("sales.csv")
con = duckdb.connect()
con.register("sales_data", df)

# Prompt template
def build_prompt(nlq):
    schema = "sales_data(region TEXT, quarter TEXT, sales INT)"
    prompt = f"""### You are an expert Postgres SQL generator.
### Given the following table schema:
# {schema}

### Write a SQL query to answer the question:
# {nlq}

### SQL:
SELECT"""
    return prompt

# Query model
def generate_sql(prompt):
    output = llm(prompt, temperature=0, max_tokens=256)
    text = output["choices"][0]["text"]

    if "SELECT" not in text.upper():
        print("❌ 'SELECT' not found in model output. Raw output:")
        print(text)
        return None

    # Try to extract SQL statement cleanly
    try:
        sql = "SELECT" + text.upper().split("SELECT", 1)[1].split(";")[0].strip() + ";"
        return sql
    except Exception as e:
        print("❌ Error while parsing SQL:", e)
        print("Raw model output:")
        print(text)
        return None

# Run query
def run_nlq(nlq):
    prompt = build_prompt(nlq)
    sql = generate_sql(prompt)

    if not sql:
        print("\n⚠️ Could not generate valid SQL.")
        return

    print("\n📜 Generated SQL:")
    print(sql)

    try:
        result = con.execute(sql).fetchdf()
        print("\n📊 Query Result:")
        print(result)
    except Exception as e:
        print("\n❌ SQL Execution Error:")
        print(e)

if __name__ == "__main__":
    run_nlq("What were the total sales in Q3 for the Northeast?")
