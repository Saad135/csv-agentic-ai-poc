import io
import json
import re
import sys
import traceback
import pandas as pd
import streamlit as st
from openai import OpenAI

# Force non-interactive Matplotlib backend for headless Streamlit server rendering
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Multi-CSV Code-as-Action Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("📂 Multi-CSV Code-as-Action Agent")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    model_choice = st.selectbox("LLM Model", ["gpt-4o", "gpt-4o-mini"], index=0)
    st.markdown("---")
    st.caption(
        "Supported Pattern: Multi-table joins, Pandas merges, Matplotlib/Seaborn charts."
    )

# File uploader allowing multiple CSV files simultaneously
uploaded_files = st.file_uploader(
    "Upload one or more CSV files", type=["csv"], accept_multiple_files=True
)


def sanitize_table_name(filename: str) -> str:
    """Sanitizes filename into a clean Python dictionary key.

    e.g., 'customer-orders 2026.csv' -> 'customer_orders_2026'
    """
    clean_name = filename.rsplit(".", 1)[0]
    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", clean_name)
    clean_name = re.sub(r"_+" , "_", clean_name).strip("_")
    return clean_name.lower()


def execute_python_code_and_capture_charts(code: str, dfs: dict):
    """Executes code against the `dfs` dictionary, capturing stdout text

    and intercepting Matplotlib/Seaborn figures from memory.
    """
    buffer = io.StringIO()
    sys.stdout = buffer

    # Clear any leftover figures in matplotlib session
    plt.close("all")

    # Expose data science libraries and the multi-CSV dictionary
    local_scope = {"dfs": dfs, "pd": pd, "plt": plt, "sns": sns}

    captured_figures = []

    try:
        exec(code, local_scope)
        output_text = buffer.getvalue().strip()

        # Intercept any active figures generated during execution
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=150)
                img_buf.seek(0)
                captured_figures.append(img_buf)

            plt.close("all")

        if not output_text and not captured_figures:
            output_text = (
                "Code executed successfully with no print output or figures generated."
            )

    except Exception:
        output_text = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = sys.__stdout__

    return output_text, captured_figures


# OpenAI Function / Tool Definition
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": (
                "Execute Python code against the dictionary of DataFrames 'dfs'. "
                "Access tables via `dfs['table_name']`. Available libraries: `dfs`, `pd`, `plt`, `sns`. "
                "You can perform pd.merge() across datasets and plot charts. Do NOT call plt.show(). "
                "Always print key figures or results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Executable Python snippet. "
                            "Example: `merged = pd.merge(dfs['orders'], dfs['customers'], on='cust_id'); print(merged.groupby('city')['amount'].sum())`"
                        ),
                    }
                },
                "required": ["code"],
            },
        },
    }
]


if uploaded_files and api_key:
    client = OpenAI(api_key=api_key)

    # 1. Load all CSVs into the `dfs` dictionary and build schema context
    dfs = {}
    schema_summaries = []

    for file in uploaded_files:
        table_name = sanitize_table_name(file.name)
        
        # Ensure unique table names if duplicate files are uploaded
        counter = 1
        original_name = table_name
        while table_name in dfs:
            table_name = f"{original_name}_{counter}"
            counter += 1

        df_temp = pd.read_csv(file)
        dfs[table_name] = df_temp

        # Structure schema metadata for the LLM
        cols_summary = dict(df_temp.dtypes)
        sample_row = (
            df_temp.head(1).to_dict(orient="records")[0]
            if not df_temp.empty
            else {}
        )

        schema_summaries.append(
            f"Table Key: `dfs['{table_name}']` (File: '{file.name}')\n"
            f"- Row Count: {len(df_temp)}\n"
            f"- Columns & Types: {cols_summary}\n"
            f"- Sample Row: {sample_row}\n"
        )

    # 2. UI Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"📊 Loaded Datasets ({len(dfs)})")
        selected_table = st.selectbox("Select dataset to preview:", list(dfs.keys()))
        if selected_table:
            st.dataframe(dfs[selected_table].head(10), use_container_width=True)
            st.caption(f"Shape: {dfs[selected_table].shape}")

        with st.expander("View Full Multi-Table Metadata"):
            for summary in schema_summaries:
                st.markdown(summary)

    with col2:
        st.subheader("💬 Ask Questions Across Your Datasets")
        user_query = st.text_input(
            "What would you like to analyze or visualize?",
            placeholder="e.g., Join orders and customers to plot revenue by customer country",
        )

        if user_query:
            st.markdown("---")
            st.subheader("⚙️ Agent Execution Trace")

            # System prompt describing the complete multi-table schema
            all_schemas_prompt = "\n---\n".join(schema_summaries)
            system_prompt = f"""
You are an expert Data Science Agent skilled at multi-table analysis using Pandas and Matplotlib/Seaborn.
You have direct access to a dictionary of DataFrames named `dfs`.

AVAILABLE DATASETS:
{all_schemas_prompt}

INSTRUCTIONS:
1. Access datasets via `dfs['table_name']`.
2. When answering queries requiring data from multiple files, use `pd.merge()`, `pd.concat()`, or cross-table lookup logic.
3. For charts, construct them using `plt`, `sns`, or `dataframe.plot()`. Do NOT call `plt.show()`.
4. ALWAYS use `print()` to output key calculations, aggregates, or final summary numbers.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ]

            max_steps = 4
            for step in range(max_steps):
                with st.spinner(f"Agent reasoning & coding (Step {step + 1})..."):
                    response = client.chat.completions.create(
                        model=model_choice,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )

                msg = response.choices[0].message
                messages.append(msg)

                # Check if the agent called the code execution tool
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.function.name == "execute_python_code":
                            args = json.loads(tool_call.function.arguments)
                            code_snippet = args.get("code", "")

                            with st.status(
                                f"Step {step + 1}: Generated Python Action",
                                expanded=True,
                            ) as status:
                                st.code(code_snippet, language="python")

                                # Execute code against the dictionary of dataframes
                                (
                                    text_output,
                                    figures,
                                ) = execute_python_code_and_capture_charts(
                                    code_snippet, dfs
                                )

                                # Render printed output
                                if text_output:
                                    st.write("**Console Output:**")
                                    st.text(text_output)

                                # Render captured charts
                                if figures:
                                    st.write("**Generated Visualizations:**")
                                    for idx, fig_buf in enumerate(figures):
                                        st.image(
                                            fig_buf,
                                            caption=f"Generated Chart {idx + 1}",
                                            use_container_width=True,
                                        )

                                status.update(
                                    label=f"Step {step + 1}: Execution Complete",
                                    state="complete",
                                )

                            # Feed execution output back into the agent's context
                            tool_feedback = f"Console Output: {text_output}\nGenerated {len(figures)} chart(s) successfully."
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_feedback,
                                }
                            )
                else:
                    # Final synthesis response from the LLM
                    st.success("Analysis Complete!")
                    st.markdown("### 🎯 Executive Summary")
                    st.write(msg.content)
                    break

elif uploaded_files and not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to proceed.")