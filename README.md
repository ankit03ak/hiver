# Grounded T-Mobile Twitter Support Agent

An assignment project for **Hiver's SDE Intern take-home**. Given a customer tweet, the
agent predicts a brand-specific intent, retrieves similar historical T-Mobile
support cases, drafts a reply, and decides whether the message can be
auto-handled or must be escalated to a human.

> 📖 **Full Assignment Report:** See [`REPORT.md`](REPORT.md) for the mandatory 6-section report including baseline comparisons, top 5 failure analysis, "What is misleading about my headline number?", and 15 non-obvious engineering decisions.

---

## ⚡ Reproduce Headline Results (Under 15 Minutes)

Run the fast evaluation harness or automated test suite from the project root:

```powershell
# 1. Activate virtual environment & install dependencies
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run fast benchmark evaluation (Simple baseline)
python -m src.evaluate --mode simple

# 3. Run fast benchmark evaluation (Grounded Gemini Agent - 5 samples)
python -m src.evaluate --mode gemini --limit 5

# 4. Run automated test suite (11 pytest tests)
python -m pytest -v
```

---

## What the project does

The application follows this flow:

1. The customer message is converted into TF-IDF word and bigram features.
2. The retriever finds the three most similar resolved T-Mobile support cases
   from the training split.
3. Deterministic risk rules look for fraud, account-security, payment,
   repeated-support, outage, and business-critical connectivity signals.
4. If `GEMINI_API_KEY` is configured, Gemini returns a structured intent,
   handling decision, and reply draft using the retrieved cases as evidence.
5. Risk flags always override the model and force escalation.
6. Without an API key, the application uses a deterministic offline fallback.

The public reply is deliberately cautious: it must not request account,
payment, PIN, identity, or other sensitive information. Account-specific
requests are directed to a secure DM/human workflow.

## Main capabilities

- Streamlit demo UI for analysing one customer message at a time.
- Offline mode that works without Gemini credentials.
- Optional Gemini-powered structured classification and reply drafting.
- Transparent TF-IDF retrieval of historical evidence.
- Eight-intent taxonomy stored in
  [`config/intent_taxonomy.json`](config/intent_taxonomy.json).
- Local Streamlit UI for blind human labelling of a golden set.
- Trivial, keyword/TF-IDF, and Gemini evaluation modes.
- Deterministic data preparation and train/test/golden splitting.

## Requirements

- Python 3.11 (recommended; Python 3.10+ should work).
- `pip`.
- Windows PowerShell, Command Prompt, macOS Terminal, or a Linux shell.
- A Gemini API key is optional. The app works in offline fallback mode without
  one.
- The raw Kaggle dataset is only needed if you want to rebuild the processed
  subset. The checked-in processed subset is sufficient to run the app.

## Installation

### Windows PowerShell

From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, either use the already-created interpreter
directly (`.\.venv\Scripts\python.exe`) or allow scripts for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

### macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

The virtual environment must be activated whenever commands are run, unless
you use its Python executable explicitly.

## Configuration

Open `.env` and set the optional Gemini values:

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

`.env` is ignored by Git. Never commit a real API key. `.env.example` is a
template only.

When `GEMINI_API_KEY` is empty or invalid, the app falls back to local
deterministic behaviour. If Gemini is unavailable during a request, the
request also falls back locally and displays the fallback mode.

## Run the Streamlit demo

From the project root with the virtual environment activated:

```powershell
streamlit run app.py
```

Then open the URL shown by Streamlit, normally
`http://localhost:8501`. Enter a customer message and select **Analyse
message**.

Example messages:

```text
My hotspot has not worked all morning and I need it for work.
```

```text
Someone opened an account in my name and I do not recognize the charge.
```

The first example is likely to be classified as a network issue. The second
contains account-security and payment risk signals and should be escalated.

To stop Streamlit, press `Ctrl+C` in the terminal.

## Project structure

```text
hiver-project/
├── app.py                         Main Streamlit support-agent UI
├── label_golden.py                Human annotation Streamlit UI
├── requirements.txt               Python dependencies
├── .env.example                   Safe environment-variable template
├── config/
│   └── intent_taxonomy.json       Allowed intents and their definitions
├── data/
│   ├── processed/
│   │   └── tmobile_pairs.csv      Reproducible compact training dataset
│   └── golden/
│       ├── annotation_template.csv Blind labels and routing decisions
│       └── reference_replies.csv  Hidden historical replies for evaluation
└── src/
    ├── agent.py                   Risk rules, fallback, and Gemini orchestration
    ├── retrieval.py               TF-IDF historical-case retrieval
    ├── prepare_data.py            Build the processed T-Mobile subset
    ├── inspect_data.py            Compare telecom-account volumes
    ├── create_golden_set.py       Create blind annotation/reference files
    └── evaluate.py                Run evaluation and write metrics
```

## Data and attribution

The source dataset is Thought Vector's
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter),
licensed CC BY-NC-SA 4.0.

The repository contains a compact derivative subset, not the full
multi-million-tweet corpus:

- 12,000 direct customer-message to T-Mobile-reply pairs.
- 11,160 training rows.
- 600 test rows.
- 240 golden-pool rows.

The split is deterministic using a SHA-256 hash of the customer tweet ID.
Retrieval uses only the training split, so test and golden rows are not used as
retrieval evidence.

## Rebuild the processed dataset

Download the Kaggle dataset manually. The input may be either `twcs.csv` or
the original ZIP; extraction is not required. Then run:

```powershell
python src/inspect_data.py --input "D:\path\to\twcs.csv.zip"
python src/prepare_data.py --input "D:\path\to\twcs.csv.zip"
```

`inspect_data.py` reports brand-authored tweet volumes for the candidate
telecom accounts. `prepare_data.py` selects direct inbound-customer to
`TMobileHelp` reply pairs and writes:

```text
data/processed/tmobile_pairs.csv
```

The default output is capped at 12,000 rows. To choose another output path or
maximum size:

```powershell
python src/prepare_data.py `
  --input "D:\path\to\twcs.csv.zip" `
  --output "data\processed\tmobile_pairs.csv" `
  --max-pairs 12000
```

## Create or reset the golden set

After rebuilding the processed data, create the blind annotation file and the
separate hidden-reply reference file:

```powershell
python src/create_golden_set.py
```

Useful options:

```powershell
python src/create_golden_set.py `
  --pairs data\processed\tmobile_pairs.csv `
  --count 200 `
  --annotation-output data\golden\annotation_template.csv `
  --reference-output data\golden\reference_replies.csv
```

The command uses a fixed random seed so the same input produces the same
golden set. Resetting the files overwrites existing annotations, so copy any
completed annotation file first if it must be preserved.

## Label the golden set

Start the annotation UI:

```powershell
streamlit run label_golden.py
```

For every row:

- Choose exactly one intent from the taxonomy.
- Choose `yes` only when the message can be handled safely without account
  access or sensitive information.
- Choose `no` when a human or secure account workflow is required.
- Provide an escalation reason whenever `no` is selected.
- Add notes for ambiguity or annotation rationale.
- Save the row to open the next example.

The UI writes progress directly to
`data/golden/annotation_template.csv`. All 200 rows are 100% completed and annotated for full benchmark evaluation.

Do not use `reference_replies.csv` while labelling intent or routing. It is
kept separate to avoid leaking historical answers into the human labels.

## Run evaluation

Evaluation requires at least one completed annotation row. For the intended
full evaluation, complete all 200 rows first.

```powershell
python -m src.evaluate --mode trivial
python -m src.evaluate --mode simple
python -m src.evaluate --mode gemini
```

What the modes mean:

- `trivial`: predicts `other` and always escalates.
- `simple`: uses deterministic keyword intent routing, risk rules, and
  one-case TF-IDF retrieval.
- `gemini`: uses the complete grounded agent, including Gemini when configured.

Limit evaluation to the first N completed rows:

```powershell
python -m src.evaluate --mode simple --limit 20
```

Use a different annotation file or output directory:

```powershell
python -m src.evaluate `
  --annotations data\golden\annotation_template.csv `
  --mode simple `
  --output-dir artifacts
```

Each run writes row-level predictions and metrics to `artifacts/`:

```text
artifacts/predictions_<mode>.csv
artifacts/metrics_<mode>.json
```

Metrics include intent accuracy, intent macro-F1, escalation precision,
escalation recall, and escalation F1.

## Quick smoke check

To verify that dependencies, the processed data, and the offline agent are
available without starting Streamlit:

```powershell
python -c "from src.agent import SupportAgent; print(SupportAgent().respond('My hotspot is not working').mode)"
```

Expected offline output contains:

```text
offline fallback
```

If a Gemini key is configured and accepted, the mode may instead contain the
configured Gemini model name.

## Intent taxonomy

The allowed labels are defined in
[`config/intent_taxonomy.json`](config/intent_taxonomy.json):

- `device_upgrade_unlock`
- `network_service`
- `billing_payment`
- `account_security`
- `plan_features_promotions`
- `order_shipping`
- `support_follow_up`
- `other`

The taxonomy is intentionally small enough for consistent human annotation.
It is provisional and should be reviewed if this project is extended beyond
the assignment.

## Important limitations

- Historical Twitter replies often direct customers to DM, so similarity may
  favour generic replies.
- The golden set is a sample and may not represent production traffic.
- There is no live account or transaction context.
- Taxonomy ambiguity can affect both intent and routing labels.
- Rare security, outage, and order cases may be hidden by aggregate metrics.
- The Gemini response is constrained by prompting and validation, not a
  production policy engine.
- There are currently no automated test files in the repository; the smoke
  check and evaluation commands are the available validation workflow.

## Attribution

- Dataset: Thought Vector, Customer Support on Twitter (CC BY-NC-SA 4.0).
- Gemini integration: [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/).
- Retrieval and metrics: [scikit-learn](https://scikit-learn.org/).
