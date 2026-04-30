<div align="center">

# 🔍 Multi Model Anomaly AI Agents

**A production-ready, plug-and-play anomaly detection system powered by a multi-agent AI pipeline**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Models-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Zero Config](https://img.shields.io/badge/Zero_Config-Any_Data-blueviolet?style=flat)

*Drop in any tabular dataset — the system auto-validates, normalizes, builds features, selects the best ML algorithm, fuses multiple signals, and generates plain-English explanations using GPT-4o.*

</div>



---

## Table of Contents

- [Overview](#overview)
- [Works on Any Data — Zero Configuration Required](#works-on-any-data--zero-configuration-required)
- [Architecture](#architecture)
- [Multi-Agent Pipeline — 10 Steps](#multi-agent-pipeline--10-steps)
  - [Agent 1 — Data Validation](#agent-1--data-validation)
  - [Agent 2 — Feature Builder](#agent-2--feature-builder)
  - [Agent 3 — Intent & Prompt Builder](#agent-3--intent--prompt-builder)
  - [Agent 4 — Detection Engine](#agent-4--detection-engine)
  - [Agent 5 — Scoring Engine](#agent-5--scoring-engine)
  - [Agent 6 — Explanation Engine](#agent-6--explanation-engine)
- [Detection Models](#detection-models)
- [Signal Fusion Formula](#signal-fusion-formula)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Setup & Run](#setup--run)
- [Environment Variables](#environment-variables)
- [UI Dashboard](#ui-dashboard)
- [Tech Stack](#tech-stack)

---

## Overview

Most anomaly detection tools require you to know your data schema upfront. This system does not. It accepts **any tabular dataset** — any columns, any types — and runs a full 10-step multi-agent pipeline automatically.

**Key capabilities:**

- Zero-config schema detection — numeric, categorical, datetime columns auto-detected
- Multi-model detection — Isolation Forest, LOF, Z-score, MAD — auto-selected per dataset
- Triple signal fusion — statistical + vector similarity + rule-based signals combined
- LLM-powered explanations — GPT-4o generates plain-English, non-technical explanations
- Offline embeddings — `all-MiniLM-L6-v2` runs fully locally, no API calls
- Session persistence — all results saved to PostgreSQL per tenant

---

## Works on Any Data — Zero Configuration Required

This is not a domain-specific anomaly detector. It does not care whether your data is about finance, healthcare, e-commerce, IoT sensors, server logs, or anything else. **The pipeline adapts itself to whatever data you give it.**

You do not need to:
- Define a schema or column types upfront
- Write any preprocessing code
- Choose a detection algorithm
- Set thresholds or sensitivity manually
- Know anything about machine learning

Just drop in your dataset — the agents figure out the rest.

**Supported data types — all handled automatically:**

| Data Type | How It Is Handled |
|---|---|
| Numeric columns | Used directly for ML detection (Z-score, LOF, Isolation Forest) |
| Categorical columns | Auto-encoded — one-hot (low cardinality) or frequency encoding (high cardinality) |
| Datetime columns | Decomposed into hour, weekday, month, is_weekend, day_of_year features |
| Boolean columns | Converted to 0/1 integers |
| Mixed columns | Each column handled independently based on its detected type |
| Missing values | KNN imputation (numeric) or mode fill (categorical) — no data loss |
| Identifier columns | Auto-detected and dropped (user_id, uuid etc.) — prevents false anomalies |
| Skewed distributions | Log-transform applied automatically before scaling |

**Real-world dataset examples this system handles out of the box:**

```
 Bank transactions      — amount, merchant, country, timestamp
 Server logs            — latency_ms, status_code, endpoint, timestamp
 E-commerce orders      — price, quantity, discount, region, category
 IoT sensor readings    — temperature, pressure, humidity, device_id
 Login / access logs    — ip_address, login_hour, failed_attempts, country
 Healthcare records     — age, blood_pressure, glucose, medication_count
 HR / employee data     — salary, department, tenure, performance_score
 Supply chain data      — lead_time, order_qty, supplier_id, delay_days
 Any CSV you have       — columns auto-detected, pipeline runs automatically
```

**The system never requires you to tell it what kind of data it is.** Agent 3 (Intent Parser) reads your optional natural language description and configures the detection focus automatically. If you provide no description at all, the system still runs — it uses all columns and auto-selects the best algorithm based on dataset size and shape.

> **In short:** If your data is tabular and has at least 10 rows with at least one numeric or datetime column — this system will find anomalies in it.

---

## Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │              FastAPI Application                     │
                        │                  (port number)                        │
                        └──────────────────────┬──────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────────┐
                    │                  10-Step Agent Pipeline                  │
                    │                                                          │
                    │  [1] Ingest → [2] Validate → [3] Normalize →            │
                    │  [4] Features → [5] Intent → [6] Embed →                │
                    │  [7] Vector → [8] Detect → [9] Score → [10] Explain     │
                    └──────────────────────────────────────────────────────────┘
                           │              │              │              │
                    ┌──────▼──────┐ ┌────▼────┐ ┌──────▼──────┐ ┌────▼──────┐
                    │  ML Models  │ │ Embedder│ │  GPT-4o LLM │ │PostgreSQL │
                    │  IF/LOF/    │ │MiniLM   │ │  Intent +   │ │ Sessions  │
                    │  Z-score    │ │L6-v2    │ │  Explain    │ │ + Alerts  │
                    │  MAD        │ │(offline)│ │             │ │           │
                    └─────────────┘ └─────────┘ └─────────────┘ └───────────┘
```

---

## Multi-Agent Pipeline — 10 Steps

Every request flows through all 10 steps in sequence. Each step is an independent agent with its own typed input/output contract (Pydantic models).

```
Input Dataset
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1  │  Ingest          │  JSON / CSV → pandas DataFrame            │
│  Step 2  │  Agent 1         │  Validate — quality gate                  │
│  Step 3  │  Normalize       │  KNN impute + RobustScaler / StandardScaler│
│  Step 4  │  Agent 2         │  Feature Builder — ML-ready matrix        │
│  Step 5  │  Agent 3         │  Intent Parser — GPT-4o intent config     │
│  Step 6  │  Embed           │  MiniLM-L6-v2 → 384-dim vectors           │
│  Step 7  │  Vector Store    │  Cosine similarity outlier scores         │
│  Step 8  │  Agent 4         │  Detection Engine — auto-select algorithm  │
│  Step 9  │  Agent 5         │  Scoring Engine — fuse 3 signals          │
│  Step 10 │  Agent 6         │  Explanation Engine — GPT-4o explanations │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
AnomalyAnalyzeResponse
```

---

### Agent 1 — Data Validation

The quality gate. If data fails critical checks, the entire pipeline halts immediately — no wasted compute.

**What it checks:**

| Check | Threshold | Action |
|---|---|---|
| Minimum rows | < 10 rows | INVALID — pipeline stops |
| No numeric/datetime columns | — | INVALID — pipeline stops |
| Overall null rate | > 50% of all cells | INVALID — pipeline stops |
| Per-column null rate | > 95% | Warning — suggest dropping |
| Per-column null rate | > 70% | Warning — will be imputed |
| Duplicate rows | > 20% of dataset | Warning — may bias results |

**Auto column type detection:**

- Numeric → used directly for ML
- Datetime → extracted into time features (hour, weekday, month, is_weekend, day_of_year)
- Categorical → one-hot encoded (low cardinality) or frequency encoded (high cardinality)

**Output:** `ValidationResult` — passed to all downstream agents

---

### Agent 2 — Feature Builder

Transforms any raw dataset into an ML-ready numeric feature matrix. Fully dynamic — no hardcoded schema.

**Pipeline (order matters):**

```
1. Drop identifier columns  (user_id, uuid — cardinality ≥ 90% of rows)
        ↓
2. Extract datetime features  (hour, weekday, month, is_weekend, day_of_year)
        ↓
3. Encode categoricals
   ├── cardinality < 20  →  one-hot encoding
   └── cardinality ≥ 20  →  frequency encoding  (avoids dimensionality explosion)
        ↓
4. Select numeric columns only
        ↓
5. Drop near-zero variance columns  (variance < 0.01 → no signal)
```

**Why identifier columns are dropped first:**

If `user_id` is one-hot encoded, every row gets its own unique column — making every row look like an outlier to LOF and Isolation Forest. Dropping identifiers before encoding prevents this false signal.

**Output:** `(feature_df, FeatureResult)` — ML-ready DataFrame + metadata

---

### Agent 3 — Intent & Prompt Builder


Translates the user's natural language request into a precise detection configuration using GPT-4o.

**Intent types:**

| Type | Triggers |
|---|---|
| `fraud` | financial transactions, payments, amounts, accounts |
| `security` | access logs, IPs, login attempts, permissions |
| `operational` | latency, errors, throughput, system metrics |
| `pattern` | time-series, sequences, behavioral patterns |
| `custom` | anything else |
| `default` | no user request provided |

**Fallback chain:**

```
User provides request
        │
        ▼
   GPT-4o parses intent  ──success──►  intent_source = "user_input"
        │
        │ fails
        ▼
   Use numeric columns only  ────────►  intent_source = "ambiguity_default"

No request provided
        │
        ▼
   Use all columns  ─────────────────►  intent_source = "default_fallback"
```

**Important design decision:** Sensitivity (contamination rate) is **never** set by the LLM. The LLM has no knowledge of the actual data distribution. Sensitivity is always `"auto"` unless the user explicitly overrides it via the API.

---

### Agent 4 — Detection Engine

Auto-selects the most appropriate anomaly detection algorithm based on 4 dimensions: dataset size, dimensionality, latency mode, and intent.

**Algorithm selection matrix:**

```
latency_mode = "realtime"
        │
        └──► Z-score  (< 50ms, single-pass, no model fitting)

latency_mode = "batch"
        │
        ├── n_rows < 50    ──► Z-score  (LOF unreliable below 50 rows)
        │
        ├── n_rows < 1000  ──► LOF  (neighborhood geometry meaningful)
        │
        └── n_rows ≥ 1000  ──► Isolation Forest  (best general-purpose)
```

**Available algorithms:**

| Algorithm | Best For | Latency |
|---|---|---|
| `isolation_forest` | Large tabular datasets, unsupervised, any dimension | Batch |
| `lof` | Small datasets (50–999 rows), density-based | Batch |
| `statistical_zscore` | Real-time prediction, < 50ms | Real-time |
| `statistical_mad` | Robust alternative to Z-score, skewed data | Real-time |

**Isolation Forest detail:**

- `contamination="auto"` → sklearn decides threshold from data distribution (recommended)
- `contamination=float` → user-specified rate (0.05–0.50)
- Rows flagged as anomaly (`-1`) get minimum score boosted to `0.70` — guarantees they cross the scoring threshold

---

### Agent 5 — Scoring Engine

Fuses 3 independent anomaly signals into a single reliable final score.

**Signal fusion formula:**

```
final_score = (0.50 × statistical_score)
            + (0.30 × vector_score)
            + (0.20 × rule_flag)
```

**Signal weights rationale:**

| Signal | Weight | Reason |
|---|---|---|
| Statistical / ML model | 0.50 | Broadest data coverage, most reliable |
| Vector similarity | 0.30 | Captures semantic / contextual anomalies |
| Rule flag | 0.20 | Domain-defined rules, highest precision |

**Anomaly determination:**

```
anomaly_flag = True   if final_score ≥ 0.65  (default threshold)
override     = True   if ANY single signal ≥ 0.90  (high-confidence override)
```

**Severity bucketing:**

| Score Range | Severity |
|---|---|
| ≥ 0.80 | `high` |
| 0.55 – 0.79 | `medium` |
| < 0.55 | `low` |

**Adaptive threshold:** When vector/rule signals are unavailable (all zero), the threshold scales down proportionally so the statistical signal alone can still flag anomalies correctly.

---

### Agent 6 — Explanation Engine

Transforms raw anomaly scores into human-readable, non-technical explanations.

**Two modes:**

| Mode | When Used | Output |
|---|---|---|
| AI mode (async) | GPT-4o available | Friendly, jargon-free sentence per anomaly |
| Fallback (sync) | LLM unavailable | Rule-based plain English |

**Design principles:**

- Explanations written for end users, not data scientists
- No sigma values, no "median", no ML terminology in output
- Each explanation answers: *"What is unusual?"* and *"Why does it matter?"*
- All flagged records explained in a **single batched LLM call** — minimizes latency and cost

**Narrative generation:**

After individual explanations, a high-level 2–3 sentence summary is generated for the entire dataset — written for a business manager, not a technical audience.

---

## Detection Models

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Detection Model Details                           │
├──────────────────────┬──────────────────────────────────────────────────┤
│  Isolation Forest    │  Ensemble of random trees. Anomalies are         │
│                      │  isolated in fewer splits. Best for large,       │
│                      │  high-dimensional tabular data.                  │
│                      │  sklearn: n_estimators=100, random_state=42      │
├──────────────────────┼──────────────────────────────────────────────────┤
│  LOF                 │  Local Outlier Factor. Compares local density     │
│  (Local Outlier      │  of a point to its neighbors. Best for small     │
│   Factor)            │  datasets where neighborhood geometry is         │
│                      │  meaningful (50–999 rows).                       │
│                      │  sklearn: n_neighbors=min(20, n_rows-1)          │
├──────────────────────┼──────────────────────────────────────────────────┤
│  Z-score             │  Per-column standard deviation distance from     │
│                      │  mean. Fastest method (< 50ms). Used for         │
│                      │  real-time prediction and very small datasets.   │
├──────────────────────┼──────────────────────────────────────────────────┤
│  MAD                 │  Median Absolute Deviation. Robust alternative   │
│  (Median Absolute    │  to Z-score. Uses median + IQR instead of        │
│   Deviation)         │  mean + std — unaffected by extreme outliers.    │
├──────────────────────┼──────────────────────────────────────────────────┤
│  MiniLM-L6-v2        │  Sentence transformer for semantic embeddings.   │
│  (Embedding Model)   │  384-dim vectors. Runs fully offline.            │
│                      │  Used for vector similarity outlier detection.   │
└──────────────────────┴──────────────────────────────────────────────────┘
```

---

## Signal Fusion Formula

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   final_score = (0.50 × stat) + (0.30 × vector) + (0.20 × rule)        │
│                                                                          │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐               │
│   │  Statistical │   │    Vector    │   │     Rule     │               │
│   │   Signal     │   │   Signal     │   │    Signal    │               │
│   │              │   │              │   │              │               │
│   │  IF / LOF /  │   │  MiniLM-L6  │   │  Vector      │               │
│   │  Z-score /   │   │  cosine      │   │  outlier     │               │
│   │  MAD         │   │  similarity  │   │  flag        │               │
│   │              │   │              │   │              │               │
│   │  weight=0.50 │   │  weight=0.30 │   │  weight=0.20 │               │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘               │
│          │                  │                  │                        │
│          └──────────────────┴──────────────────┘                        │
│                             │                                            │
│                             ▼                                            │
│                      final_score (0.0 – 1.0)                            │
│                             │                                            │
│              ┌──────────────┼──────────────┐                            │
│              │              │              │                            │
│           ≥ 0.80         0.55–0.79       < 0.55                        │
│              │              │              │                            │
│            HIGH          MEDIUM           LOW                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
multi_model_anomoly_ai_agents/
│
├── src/                              # Application source code
│   │
│   ├── main.py                       # FastAPI app entry point
│   ├── middleware.py                 # Vault secret middleware (tenant DB pools)
│   ├── secret_proxy.py               # HashiCorp Vault secret client
│   │
│   ├── config/                       # Configuration
│   │   ├── __init__.py               # Loads settings based on ENV
│   │   ├── config_local.py           # Local development settings
│   │   └── auth.py                   # JWT auth dependency
│   │
│   ├── routes/                       # API route handlers
│   │   └── anomaly_routes.py         # All 5 anomaly endpoints
│   │
│   ├── schemas/                      # Pydantic request/response models
│   │   └── anomaly_schemas.py        # All typed contracts
│   │
│   ├── services/                     # Core agent implementations
│   │   │
│   │   ├── ingestion.py              # Step 1  — JSON/CSV → DataFrame
│   │   ├── validation_agent.py       # Step 2  — Agent 1: Data Validation
│   │   ├── normalization.py          # Step 3  — KNN impute + scaling
│   │   ├── feature_builder.py        # Step 4  — Agent 2: Feature Builder
│   │   ├── prompt_builder.py         # Step 5  — Agent 3: Intent Parser (GPT-4o)
│   │   ├── embedding_layer.py        # Step 6  — MiniLM-L6-v2 embeddings
│   │   ├── embedding_store.py        # Step 6b — Save embeddings to PostgreSQL
│   │   ├── vector_store.py           # Step 7  — Cosine similarity outlier scores
│   │   ├── detection_engine.py       # Step 8  — Agent 4: Detection Engine
│   │   ├── scoring_engine.py         # Step 9  — Agent 5: Signal Fusion
│   │   ├── explanation_engine.py     # Step 10 — Agent 6: Explanation Engine
│   │   ├── result_store.py           # Save sessions + alerts to PostgreSQL
│   │   └── llm_client.py             # OpenAI GPT-4o client
│   │
│   ├── db/                           # Database layer
│   │   ├── session.py                # asyncpg connection pool management
│   │   ├── models.py                 # Table definitions
│   │   └── migrations/               # SQL migration files
│   │
│   └── utils/
│       └── vault_secrets.py          # Vault secret utilities
│
├── models/                           # Pre-downloaded embedding model
│   └── all-MiniLM-L6-v2/            # sentence-transformers model (offline)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── vocab.txt


```

---

## API Endpoints

Base URL: `http://localhost:port`

### POST `/api/v1/anomaly/analyze`

Run the full 10-step pipeline on a JSON dataset.

**Request:**
```json
{
  "data": {
    "columns": ["amount", "duration", "score"],
    "rows": [
      [120, 5, 0.8],
      [130, 6, 0.75],
      [9999, 1, 0.1]
    ]
  },
  "user_request": "detect unusual transactions",
  "narrate": true,
  "sensitivity": null
}
```

**Response:**
```json
{
  "status": "success",
  "total_rows": 3,
  "anomalies_found": 1,
  "anomaly_rate": 0.3333,
  "algorithm_used": "statistical_zscore",
  "model_used": "openai/gpt-4o",
  "narrative": "One record stood out with an unusually large amount...",
  "anomalies": [
    {
      "row_index": 2,
      "anomaly_score": 0.92,
      "anomaly_flag": true,
      "severity": "high",
      "top_features": ["amount"],
      "feature_values": {"amount": 9999},
      "explanation_text": "This transaction has an unusually large amount compared to all others.",
      "signal_breakdown": {"stat": 0.95, "vector": 0.0, "rule": 0.0},
      "signal_sources": ["statistical"]
    }
  ]
}
```

---

### POST `/api/v1/anomaly/predict`

Real-time single-record prediction (< 50ms, Z-score).

**Request:**
```json
{
  "record": {"amount": 9999, "duration": 1, "score": 0.1},
  "reference_data": {
    "columns": ["amount", "duration", "score"],
    "rows": [[120, 5, 0.8], [130, 6, 0.75]]
  }
}
```

---

### POST `/api/v1/anomaly/upload-csv`

Upload a CSV file — same full pipeline as `/analyze`.

**Form fields:** `file` (CSV), `user_request`, `narrate`, `sensitivity`

---

### GET `/api/v1/anomaly/sessions/{session_id}`

Fetch all saved anomaly alerts for a specific session.

---

### GET `/api/v1/anomaly/sessions`

List all sessions with stats and top alert previews.

**Query params:** `limit`, `offset`, `intent_type`, `min_anomaly_rate`

---

## Setup & Run

### 1. Clone & install dependencies

```bash
cd multi_model_anomoly_ai_agents
pip install uv
uv sync
```

### 2. Create `.env`

```bash
# Required for LLM features (intent parsing + explanations)
OPENAI_API_KEY=your_openai_api_key_here

# Optional — leave empty to disable DB persistence
DB_SCHEMA=anomaly
DB_CREDS_PATH=kv/data/anomaly-service

# Auth
JWT_SECRET_KEY=your_secret_key_here
```

### 3. Start the server

```bash
cd src
uvicorn main:app --host 0.0.0.0 --port your_port --reload
```


### 4. Run with Docker

```bash
docker build -t anomaly-detection .
docker run -p 9005:9005 --env-file .env.local anomaly-detection
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes (for LLM) | `""` | GPT-4o API key — leave empty to disable LLM features |
| `OPENAI_MODEL` | No | `gpt-4o` | OpenAI model name |
| `EMBEDDING_MODEL` | No | `models/all-MiniLM-L6-v2` | Path to local embedding model |
| `JWT_SECRET_KEY` | No | `""` | JWT signing secret |
| `DB_SCHEMA` | No | `anomaly` | PostgreSQL schema name |
| `DB_CREDS_PATH` | No | `kv/data/anomaly-service` | Vault path for DB credentials |
| `REDIS_HOST` | No | `""` | Redis host |
| `REDIS_PORT` | No | `--` | Redis port |
| `ENV` | No | `local` | Environment name |

> **Note:** If `OPENAI_API_KEY` is empty, the system still works — intent parsing falls back to `default_fallback` and explanations use the rule-based fallback engine. No LLM calls are made.

---


## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| ML Detection | scikit-learn (Isolation Forest, LOF) |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 (offline) |
| LLM | OpenAI GPT-4o |
| Data Processing | pandas, numpy |
| Normalization | scikit-learn (RobustScaler, StandardScaler, KNNImputer) |
| Database | PostgreSQL via asyncpg |
| Secrets | HashiCorp Vault |
| Validation | Pydantic v2 |
| Logging | structlog |
| Container | Docker |
| Python | 3.11+ |

---
## Author

**Shivam Sharma**
