"""The demo app for the workshop. Start it with:  streamlit run app.py"""

import pandas as pd
import streamlit as st

from budget_agent import config

st.set_page_config(page_title="Agent on a Budget", page_icon="🛒", layout="wide")
st.title("🛒 NovaMart support: agent on a budget")

LAYER_LABEL = {"ml": "🟢 ML model", "local_llm": "🟡 Local LLM", "cloud_llm": "🔵 Cloud LLM agent"}


@st.cache_resource
def get_bot(version):
    if version == 4:
        from budget_agent.router import Router
        return Router()
    from budget_agent.agent import SupportAgent
    return SupportAgent(use_tools=(version == 2))


chat_tab, compare_tab = st.tabs(["💬 Chat", "📊 Version 2 vs Version 4"])

with chat_tab:
    version = st.radio("Which version?", [1, 2, 4], horizontal=True, index=2, format_func=lambda v: {
        1: "V1: LLM only", 2: "V2: LLM + tools", 4: "V4: router (ML → local LLM → cloud LLM)"}[v])

    if "history" not in st.session_state:
        st.session_state.history = []
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.write(turn["text"])
            if "info" in turn:
                st.caption(turn["info"])

    if text := st.chat_input("Write a customer message, e.g. 'where is my order NM10005?'"):
        st.session_state.history.append({"role": "user", "text": text})
        bot = get_bot(version)
        with st.spinner("Thinking..."):
            if version == 4:
                r = bot.handle(text)
                info = (f"{LAYER_LABEL[r.layer]} · intent: {r.intent} ({r.confidence:.0%}) · "
                        f"{r.seconds:.2f}s · ${r.cost_usd:.4f}")
            else:
                r = bot.run(text)
                tools = ", ".join(name for name, _, _ in r.tool_calls) or "none"
                info = f"V{version} · tools: {tools} · {r.seconds:.2f}s · ${r.cost_usd:.4f}"
        st.session_state.history.append({"role": "assistant", "text": r.reply, "info": info})
        st.rerun()

with compare_tab:
    results_file = config.DATA_DIR / "benchmark_results.csv"
    if not results_file.exists():
        st.info("No results yet. Run `python run.py benchmark --n 30` in the terminal first.")
    else:
        results = pd.read_csv(results_file)
        cost_v2, cost_v4 = results.v2_cost.sum(), results.v4_cost.sum()
        a, b, c = st.columns(3)
        a.metric("Messages", len(results))
        b.metric("Cost: V2 → V4", f"${cost_v4:.3f}", f"{cost_v4 - cost_v2:+.3f} vs V2", delta_color="inverse")
        c.metric("Average time: V2 → V4", f"{results.v4_seconds.mean():.2f}s",
                 f"{results.v4_seconds.mean() - results.v2_seconds.mean():+.2f}s", delta_color="inverse")

        left, right = st.columns(2)
        left.subheader("Who answered in V4?")
        left.bar_chart(results.v4_layer.map(LAYER_LABEL).value_counts())
        right.subheader("Seconds per message")
        right.bar_chart(pd.DataFrame({"V2": results.v2_seconds, "V4": results.v4_seconds}))

        st.subheader("Every message")
        st.dataframe(results[["message", "v4_layer", "v2_seconds", "v4_seconds", "v2_cost", "v4_cost",
                              "v2_reply", "v4_reply"]], use_container_width=True)
