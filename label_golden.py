"""Small local UI for human annotation of the held-out golden set."""

from pathlib import Path
import json

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "data" / "golden" / "annotation_template.csv"
INTENTS = list(json.loads((ROOT / "config" / "intent_taxonomy.json").read_text(encoding="utf-8")))

st.set_page_config(page_title="Golden-set labelling", page_icon="🏷️", layout="wide")
st.title("Golden-set labelling")
st.caption("Label the customer message only. Do not look up the historical reply while assigning intent or routing.")

labels = pd.read_csv(ANNOTATIONS, keep_default_na=False)
remaining = labels.index[labels["annotation_status"].ne("complete")].tolist()
st.progress((len(labels) - len(remaining)) / len(labels), text=f"{len(labels) - len(remaining)} / {len(labels)} complete")

if not remaining:
    st.success("All labels are complete.")
    st.stop()

index = remaining[0]
row = labels.loc[index]
st.subheader(row["example_id"])
st.info(row["customer_text"])

with st.form("annotation"):
    # Per-example keys prevent a previous row's widget state leaking into the next row.
    intent = st.selectbox(
        "Intent",
        INTENTS,
        index=INTENTS.index(row["intent"]) if row["intent"] in INTENTS else 0,
        key=f"intent_{row['example_id']}",
    )
    auto_handle = st.radio(
        "Can it be auto-handled safely?",
        ["yes", "no"],
        index=0 if row["auto_handle"] != "no" else 1,
        key=f"auto_handle_{row['example_id']}",
    )
    escalation_reason = st.text_input(
        "Escalation reason (required when no)",
        value=row["escalation_reason"],
        key=f"reason_{row['example_id']}",
    )
    notes = st.text_area("Notes / ambiguity", value=row["notes"], key=f"notes_{row['example_id']}")
    submit = st.form_submit_button("Save and open next")

if submit:
    if auto_handle == "no" and not escalation_reason.strip():
        st.error("Give a short escalation reason for a message that cannot be auto-handled.")
    else:
        labels.loc[index, ["intent", "auto_handle", "escalation_reason", "notes", "annotation_status"]] = [
            intent,
            auto_handle,
            escalation_reason.strip(),
            notes.strip(),
            "complete",
        ]
        labels.to_csv(ANNOTATIONS, index=False)
        st.rerun()
