"""The demo app for the workshop. Start it with:  streamlit run app.py"""

import pandas as pd
import streamlit as st

from budget_agent import config

st.set_page_config(page_title="Agent on a Budget", page_icon="🛒", layout="wide")
st.title("🛒 NovaMart support: agent on a budget")

ICONS = {"ml": "🟢", "local_llm": "🟡", "cloud_llm": "🔵", "human": "🔴"}
LAYER_LABEL = {layer: f"{ICONS[layer]} {name}" for layer, name in config.HELPERS.items()}


@st.cache_resource
def get_bot(version):
    if version == 4:
        from budget_agent.router import Router
        return Router()
    from budget_agent.agent import SupportAgent
    return SupportAgent(use_tools=(version == 2))


chat_tab, compare_tab = st.tabs(["💬 Chat", "📊 Expert only vs the whole office"])

with chat_tab:
    version = st.radio("Which step?", [1, 2, 4], horizontal=True, index=2, format_func=lambda v: {
        1: "Chapter 1a: LLM only (can talk)",
        2: "Chapter 1b: LLM + tools (the expert)",
        4: "Chapter 3: the whole support office"}[v])

    st.caption("Try one example per helper:")
    clicked = None
    for column, (layer, message) in zip(st.columns(4), config.DEMO_MESSAGES.items()):
        if column.button(LAYER_LABEL[layer], help=message, use_container_width=True):
            clicked = message

    if "history" not in st.session_state:
        st.session_state.history = []
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.write(turn["text"])
            if "info" in turn:
                st.caption(turn["info"])

    text = st.chat_input("Write a customer message, e.g. 'track my order NM10009'") or clicked
    if text:
        st.session_state.history.append({"role": "user", "text": text})
        bot = get_bot(version)
        with st.spinner("Thinking..."):
            if version == 4:
                r = bot.handle(text)
                helper = LAYER_LABEL[r.layer]
                if r.escalated:
                    helper += f" → {LAYER_LABEL['human']}"
                info = (f"{helper} · intent: {r.intent} ({r.confidence:.0%}) · "
                        f"{r.seconds:.2f}s · ${r.cost_usd:.4f}")
            else:
                r = bot.run(text)
                tools = ", ".join(name for name, _, _ in r.tool_calls) or "none"
                info = f"tools used: {tools} · {r.seconds:.2f}s · ${r.cost_usd:.4f}"
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
        b.metric("Cost: whole office", f"${cost_v4:.3f}", f"{cost_v4 - cost_v2:+.3f} vs expert only", delta_color="inverse")
        c.metric("Average time: whole office", f"{results.v4_seconds.mean():.2f}s",
                 f"{results.v4_seconds.mean() - results.v2_seconds.mean():+.2f}s vs expert only", delta_color="inverse")

        left, right = st.columns(2)
        left.subheader("Who answered?")
        left.bar_chart(results.v4_layer.map(LAYER_LABEL).value_counts())
        right.subheader("Seconds per message")
        right.bar_chart(pd.DataFrame({"Expert only": results.v2_seconds, "Whole office": results.v4_seconds}))

        st.subheader("Every message")
        st.dataframe(results[["message", "v4_layer", "v2_seconds", "v4_seconds", "v2_cost", "v4_cost",
                              "v2_reply", "v4_reply"]], use_container_width=True)
