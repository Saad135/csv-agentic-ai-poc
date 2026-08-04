import io
import json
import re
import sys
import traceback

# Set non-interactive Matplotlib backend for Streamlit server rendering
import matplotlib
import pandas as pd
import streamlit as st
from openai import OpenAI

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
    """Execute code locally against `dfs`, capture stdout text,

    and intercept Matplotlib/Seaborn figures as raw bytes.
    """
    buffer = io.StringIO()
    sys.stdout = buffer

    plt.close("all")
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
                captured_figures.append(img_buf.getvalue())  # Store raw bytes
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
                "Always print key statistics and summary findings."
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
# Session State Initialization & Data Pre-Processing
# -------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

dfs = {}
schema_summaries = []

if uploaded_files:
    for file in uploaded_files:
        table_name = sanitize_table_name(file.name)

        # Handle duplicate file names gracefully
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

    # Show dataset inspector in the sidebar
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
        # If assistant message has execution trace steps, render them in expandable status blocks
        if "steps" in msg and msg["steps"]:
            for step_idx, step in enumerate(msg["steps"]):
                with st.expander(f"🛠️ Execution Step {step_idx + 1}", expanded=False):
                    st.code(step["code"], language="python")
                    if step["output"]:
                        st.text(f"Console Output:\n{step['output']}")
                    if step["charts"]:
                        for chart_bytes in step["charts"]:
                            st.image(chart_bytes, use_container_width=True)

        # Render final markdown text message
        if msg["content"]:
            st.markdown(msg["content"])

# -------------------------------------------------------------------
# User Input & Agent Loop
# -------------------------------------------------------------------
user_input = st.chat_input(
    "Ask a question or request a visualization across your datasets..."
)

if user_input:
    # Validation checks
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one CSV file in the sidebar.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # 1. Display User Message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Build System Prompt with current multi-table schema
    all_schemas_prompt = "\n---\n".join(schema_summaries)
    system_prompt = f"""
You are an expert AI Data Analyst conversing with a user in a chatbot interface.
You have direct access to a dictionary of DataFrames named `dfs`.

AVAILABLE DATASETS:
{all_schemas_prompt}

INSTRUCTIONS:
1. Access datasets via `dfs['table_name']`.
2. When answering multi-table questions, use `pd.merge()`, `pd.concat()`, or cross-table filter logic.
3. Construct charts using `plt`, `sns`, or `df.plot()`. Do NOT call `plt.show()`.
4. Always print key numerical figures and summarize your findings cleanly in your final message response.
"""

    # Build conversation context for API call from session state
    llm_messages = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages:
        llm_messages.append({"role": m["role"], "content": m["content"]})

    # 3. Stream Assistant Chat Turn
    with st.chat_message("assistant"):
        assistant_steps = []
        final_response_text = ""

        max_steps = 4
        for step in range(max_steps):
            with st.spinner(f"Analyzing & Coding (Step {step + 1})..."):
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=llm_messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )

            response_msg = response.choices[0].message
            llm_messages.append(response_msg)

            # Handle Code Action Tool Calls
            if response_msg.tool_calls:
                for tool_call in response_msg.tool_calls:
                    if tool_call.function.name == "execute_python_code":
                        args = json.loads(tool_call.function.arguments)
                        code_snippet = args.get("code", "")

                        # Render live step execution inside expandable status block
                        with st.status(
                            f"Step {step + 1}: Executing Code Action",
                            expanded=True,
                        ) as status:
                            st.code(code_snippet, language="python")

                            text_output, charts = (
                                execute_python_code_and_capture_charts(
                                    code_snippet, dfs
                                )
                            )

                            if text_output:
                                st.write("**Console Output:**")
                                st.text(text_output)

                            if charts:
                                st.write("**Generated Charts:**")
                                for chart_bytes in charts:
                                    st.image(
                                        chart_bytes,
                                        use_container_width=True,
                                    )

                            status.update(
                                label=f"Step {step + 1}: Complete",
                                state="complete",
                            )

                        # Save step data to persist in chat history
                        assistant_steps.append(
                            {
                                "code": code_snippet,
                                "output": text_output,
                                "charts": charts,
                            }
                        )

                        # Feed output back into conversation history
                        tool_feedback = f"Console Output: {text_output}\nGenerated {len(charts)} chart(s)."
                        llm_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_feedback,
                            }
                        )
            else:
                # Final text response from Assistant
                final_response_text = response_msg.content or ""
                if final_response_text:
                    st.markdown(final_response_text)
                break

        # If no final response was generated, request one
        if not final_response_text and assistant_steps:
            with st.spinner("Generating final summary..."):
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=llm_messages,
                    tools=TOOLS,
                    tool_choice="none",
                )
                final_response_text = response.choices[0].message.content or ""
                if final_response_text:
                    st.markdown(final_response_text)

        # Save assistant turn to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_response_text,
                "steps": assistant_steps,
            }
        )
