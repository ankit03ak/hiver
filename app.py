import streamlit as st

from src.agent import SupportAgent


st.set_page_config(page_title="T-Mobile Support Agent", page_icon="📱", layout="wide")
st.title("Grounded T-Mobile Support Agent")
st.caption("Retrieves similar historical support cases before drafting. High-risk messages are escalated.")

if "agent" not in st.session_state:
    st.session_state.agent = SupportAgent()

message = st.text_area("Customer message", placeholder="My hotspot has not worked all morning and I need it for work.", height=120)
if st.button("Analyse message", type="primary", disabled=not message.strip()):
    result = st.session_state.agent.respond(message)
    left, right = st.columns(2)
    left.metric("Intent", result.intent.replace("_", " ").title())
    right.metric("Decision", "Escalate to human" if result.handling == "escalate" else "Auto-handle")
    st.write("**Reason:**", result.escalation_reason)
    st.write("**Draft reply:**")
    st.info(result.reply)
    st.caption(f"Mode: {result.mode}")
    st.subheader("Historical evidence")
    for number, case in enumerate(result.evidence, 1):
        with st.expander(f"Case {number} — similarity {case.score:.2f}"):
            st.write("**Customer:**", case.customer_text)
            st.write("**Historical reply:**", case.historical_reply)
