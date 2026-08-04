import ast
import io
import json
import re
import sys
import traceback

import matplotlib
import pandas as pd
import streamlit as st
from openai import OpenAI

# Set non-interactive Matplotlib backend for Streamlit server rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Page Setup
st.set_page_config(
    page_title="CSV Data Analyst Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💬 Multi-CSV AI Data Analyst")


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def sanitize_table_name(filename: str) -> str:
    """Sanitize filename into a clean Python dictionary key."""
    clean_name = filename.rsplit(".", 1)[0]
    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", clean_name)
    clean_name = re.sub(r"_+", "_", clean_name).strip("_")
    return clean_name.lower()


def execute_python_code_and_capture_charts(code: str, dfs: dict):
    """Execute code locally against `dfs`, capture stdout text (with AST auto-print),

    and intercept Matplotlib/Seaborn figures.
    """
    buffer = io.StringIO()
    sys.stdout = buffer

    plt.close("all")
    local_scope = {"dfs": dfs, "pd": pd, "plt": plt, "sns": sns}
    captured_figures = []

    try:
        # Try AST modification to automatically wrap the last expression in print()
        # (e.g., converts `df.head()` to `print(df.head())`)
        try:
            tree = ast.parse(code)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_val = tree.body[-1].value
                # Avoid wrapping if it's already a print statement
                is_print = (
                    isinstance(last_val, ast.Call)
                    and isinstance(last_val.func, ast.Name)
                    and last_val.func.id == "print"
                )
                if not is_print:
                    tree.body[-1] = ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="print", ctx=ast.Load()),
                            args=[last_val],
                            keywords=[],
                        )
                    )
                    ast.fix_missing_locations(tree)
            compiled_code = compile(tree, filename="<ast>", mode="exec")
            exec(compiled_code, local_scope)
        except Exception:
            # Fallback to direct execution if AST parsing fails
            exec(code, local_scope)

        output_text = buffer.getvalue().strip()

        # Intercept any active figures generated during execution
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=150)
                captured_figures.append(img_buf.getvalue())
            plt.close("all")

        if not output_text and not captured_figures:
            output_text = "Code executed successfully with no returned output or figures generated."

    except Exception:
        output_text = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = sys.__stdout__

    return output_text, captured_figures


# Tool definition for OpenAI Tool Calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": (
                "Execute Python code against the dictionary of DataFrames `dfs`. "
                "Access datasets via `dfs['table_name']`. Available packages: `dfs`, `pd`, `plt`, `sns`. "
                "Use pd.merge() for multi-table queries. Do NOT call plt.show(). "
                "Always print key statistics and results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Executable Python code snippet.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

# -------------------------------------------------------------------
# Sidebar: Configuration & Data Management
# -------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    model_choice = st.selectbox("LLM Model", ["gpt-4o", "gpt-4o-mini"], index=0)

    st.markdown("---")
    st.header("📂 Data Upload")
    uploaded_files = st.file_uploader(
        "Upload CSV files", type=["csv"], accept_multiple_files=True
    )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# -------------------------------------------------------------------
# Session State & Dataset Initialization
# -------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

dfs = {}
schema_summaries = []

if uploaded_files:
    for file in uploaded_files:
        table_name = sanitize_table_name(file.name)

        counter = 1
        original_name = table_name
        while table_name in dfs:
            table_name = f"{original_name}_{counter}"
            counter += 1

        df_temp = pd.read_csv(file)
        dfs[table_name] = df_temp

        cols_summary = dict(df_temp.dtypes)
        sample_row = (
            df_temp.head(1).to_dict(orient="records")[0] if not df_temp.empty else {}
        )

        schema_summaries.append(
            f"Table Key: `dfs['{table_name}']` (File: '{file.name}')\n"
            f"- Row Count: {len(df_temp)}\n"
            f"- Columns & Types: {cols_summary}\n"
            f"- Sample Row: {sample_row}\n"
        )

    with st.sidebar:
        st.markdown("---")
        st.subheader(f"📊 Datasets Preview ({len(dfs)})")
        selected_table = st.selectbox(
            "Inspect Dataset:", list(dfs.keys()), key="preview_select"
        )
        if selected_table:
            st.dataframe(dfs[selected_table].head(5), use_container_width=True)
            st.caption(f"Shape: {dfs[selected_table].shape}")

# -------------------------------------------------------------------
# Chat History Rendering
# -------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Render code execution traces, output text, and charts directly
        if "steps" in msg and msg["steps"]:
            for step_idx, step in enumerate(msg["steps"]):
                st.markdown(f"**Code Execution (Step {step_idx + 1}):**")
                st.code(step["code"], language="python")

                if step["output"]:
                    st.markdown("**Output Result:**")
                    st.text(step["output"])

                if step["charts"]:
                    for chart_bytes in step["charts"]:
                        st.image(chart_bytes, use_container_width=True)

        # Render final markdown text message
        if msg["content"]:
            st.markdown(msg["content"])

# -------------------------------------------------------------------
# User Input & Execution Loop
# -------------------------------------------------------------------
user_input = st.chat_input("Ask a question about your datasets...")

if user_input:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one CSV file in the sidebar.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Prepare system prompt
    all_schemas_prompt = "\n---\n".join(schema_summaries)
    system_prompt = f"""
You are an expert AI Data Analyst conversing in a chatbot interface.
You have access to a dictionary of DataFrames named `dfs`.

AVAILABLE DATASETS:
{all_schemas_prompt}

INSTRUCTIONS:
1. Access datasets via `dfs['table_name']`.
2. When answering queries, generate Python code to compute metrics or generate plots.
3. ALWAYS print key output, dataframes, or numbers explicitly.
4. Synthesize and clearly present the calculated results in your final text response.
"""

    llm_messages = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages:
        llm_messages.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        assistant_steps = []
        final_response_text = ""

        max_steps = 4
        for step in range(max_steps):
            with st.spinner(f"Analyzing... (Step {step + 1})"):
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=llm_messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )

            response_msg = response.choices[0].message
            llm_messages.append(response_msg)

            if response_msg.tool_calls:
                for tool_call in response_msg.tool_calls:
                    if tool_call.function.name == "execute_python_code":
                        args = json.loads(tool_call.function.arguments)
                        code_snippet = args.get("code", "")

                        # Render code snippet immediately
                        st.markdown(f"**Code Execution (Step {step + 1}):**")
                        st.code(code_snippet, language="python")

                        # Run code and capture outputs/charts
                        text_output, charts = execute_python_code_and_capture_charts(
                            code_snippet, dfs
                        )

                        # Display execution output directly in the chat window
                        if text_output:
                            st.markdown("**Output Result:**")
                            st.text(text_output)

                        if charts:
                            for chart_bytes in charts:
                                st.image(chart_bytes, use_container_width=True)

                        assistant_steps.append(
                            {
                                "code": code_snippet,
                                "output": text_output,
                                "charts": charts,
                            }
                        )

                        tool_feedback = (
                            f"Console Output:\n{text_output}\n"
                            f"Generated {len(charts)} chart(s)."
                        )
                        llm_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_feedback,
                            }
                        )
            else:
                final_response_text = response_msg.content
                st.markdown(final_response_text)
                break

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_response_text,
                "steps": assistant_steps,
            }
        )
