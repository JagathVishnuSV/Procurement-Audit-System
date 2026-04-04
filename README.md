# Intelligent Procurement Audit & Anomaly Detection System

Enterprise-style procurement intelligence platform for anomaly detection, AI-assisted triage, contract-aware forensic analysis, and measurable remediation outcomes.

## Executive Summary

This platform is built to answer high-value procurement risk questions:
- Which transactions are statistically anomalous?
- Which anomalies should be escalated for deeper legal/forensic review?
- Which contract clauses might be implicated?
- Which remediation actions should be tracked to closure?
- What financial impact has been recovered?

The design intentionally separates fast screening from deeper investigation:
- **Fast path**: scoring and case handling at scale.
- **Deep path**: richer AI/legal reasoning only when escalation is warranted.

This balances cost, speed, and evidence quality.

## Architecture at a Glance

### Backend
- **Framework**: FastAPI
- **Database**: PostgreSQL (system of record)
- **Streaming backbone**: Redpanda/Kafka
- **ML**: Isolation Forest + SHAP
- **Contract intelligence**: RAG over FAISS + sentence-transformers
- **AI orchestration**: Stage 1 triage -> Stage 2 deep audit only on escalation
- **Realtime UX feed**: websocket snapshot stream

### Frontend
- **Stack**: React + Vite + TypeScript
- **Data layer**: React Query + typed API client
- **Views**: Dashboard, Audit Inbox, Forensic Workspace, Smart CLM

## Multi-Source Procurement Ingestion (NEW)

To avoid single-dataset bias and improve enterprise realism, ingestion now supports multiple procurement ecosystems.

### Integrated source families
- **USAspending** (baseline structured federal data)
- **India CPPP** (messy, semi-structured tenders)
- **Open Contracting / OCDS** (global standardized releases)

### Live source links/endpoints (connected)
- **USAspending API**: `https://api.usaspending.gov` (via existing API client)
- **India CPPP active tenders**: `https://eprocure.gov.in/eprocure/app?page=FrontEndLatestActiveTenders&service=page`
- **Open Contracting (UK Contracts Finder OCDS JSONL feed)**: `https://data.open-contracting.org/en/publication/128/download?name=2026.jsonl.gz`

### Adapter framework (implemented)
`backend/ingestion/multi_source.py` includes:
- `class BaseIngestionAdapter`
  - `fetch()`
  - `normalize()`
- `class USASpendingAdapter`
- `class CPPPAdapter`
- `class OCDSAdapter`

This provides a stable extension point for adding future sources without rewriting downstream scoring/orchestration.

### Canonical normalization schema (implemented)
All adapters normalize into one canonical structure:

```json
{
  "transaction_id": "...",
  "buyer": "...",
  "vendor": "...",
  "amount": 0,
  "currency": "...",
  "timestamp": "...",
  "source": "CPPP | USA | OCDS"
}
```

### Data quality layer (implemented)
`backend/ingestion/data_quality.py` provides:
- missing-field fallback handling
- currency normalization (with USD conversion support)
- duplicate detection (transaction ID + fingerprint strategy)
- inconsistent naming cleanup for buyers/vendors

Without this layer, mixed-source anomaly signals degrade quickly.

## Near Real-Time Ingestion Pipeline (NEW)

Realtime output is now paired with a proper ingestion backbone.

`backend/ingestion/realtime_pipeline.py` implements:

```text
[Sources]
   ↓
Pollers (5–10 min interval)
   ↓
Change detection (new/updated tenders)
   ↓
Redpanda/Kafka
   ↓
Consumers (scoring + orchestration)
```

### Core behavior
- source pollers fetch per adapter
- canonical normalization + data quality pass
- change detector emits only new/changed records
- records are published to Kafka (`raw_transactions`) for downstream processing

This upgrades Redpanda from passive infrastructure to active ingestion backbone.

## Entity Resolution Layer (NEW)

Vendor identity is now treated as an intelligence problem, not a raw string field.

`backend/ml/entity_resolution.py` implements `class EntityResolver` with:
- vendor text normalization
- fuzzy matching (`SequenceMatcher` ratio)
- vendor clustering / canonical mapping

Example outcome:
- `ABC Ltd`
- `ABC Pvt Ltd`
- `A.B.C Limited`

→ same canonical vendor cluster.

This is required for cross-record behavior tracking and collusion analytics.

## Relationship Graph Engine (NEW)

Row-level anomaly scoring is retained, but now supplemented by network-level detection.

`backend/ml/relationship_graph_engine.py` implements `class RelationshipGraphEngine` with graph modeling:
- `vendor ↔ buyer`
- `vendor ↔ vendor` (shared buyer relationships)
- `buyer ↔ category`

Detected signal families:
- repeated awards
- tight clusters
- unusual connections (centrality-based)

This surfaces suspicious network patterns, not just isolated outlier rows.

## Existing Delivery Coverage

### Sprint 1 — Data Foundation ✅
- Infra stack and persistence baseline.
- Production-style ingestion workflow.

### Sprint 2 — ML Scoring ✅
- Isolation Forest scoring endpoints.
- SHAP explainability integration.

### Sprint 3 — Contract Intelligence ✅
- Contract ingestion/chunking/indexing.
- Semantic clause retrieval.

### Sprint 4 — AI Orchestration ✅
- Triage + gated deep audit.
- Retry/rate/quota-aware handling.
- Persisted audit evidence in cases.

### Sprint 5 — Operational UX + KPIs ✅
- Case workspace + action plans.
- ROI/coverage metrics.
- Near-realtime dashboard/timeline.

## API Surface

### Health
- `GET /api/v1/health`

### Scoring
- `POST /api/v1/score`
- `POST /api/v1/score/batch`

### Contracts / CLM
- `POST /api/v1/contracts/upload`
- `GET /api/v1/contracts`
- `GET /api/v1/contracts/{contract_id}`
- `GET /api/v1/contracts/search?q=...`
- `DELETE /api/v1/contracts/{contract_id}`

### Cases
- `GET /api/v1/cases`
- `GET /api/v1/cases/{case_id}`
- `PATCH /api/v1/cases/{case_id}/status`
- `PATCH /api/v1/cases/{case_id}/notes`

### Audit orchestration
- `POST /api/v1/audit/trigger/{transaction_id}`
- `POST /api/v1/orchestration/run`
- `GET /api/v1/orchestration/status`

### Action plans
- `POST /api/v1/cases/{case_id}/action-plan`
- `GET /api/v1/action-plans`
- `PATCH /api/v1/action-plans/{plan_id}/status`

### Metrics
- `GET /api/v1/metrics/roi`
- `GET /api/v1/metrics/coverage`

### Realtime
- `WS /api/v1/realtime/stream`

## Runbook

### 1) Start infrastructure
```powershell
docker-compose up -d
```

### 2) Start backend
```powershell
.\venv\Scripts\Activate.ps1
python -m alembic upgrade head
python -m uvicorn backend.api.main:app --reload --port 8000
```

### 3) Start frontend
```powershell
cd frontend
npm install
npm run dev
```

Open: `http://localhost:5173`

### 4) Run near-realtime multi-source ingestion (new)
```powershell
python -m backend.ingestion.run_realtime_pipeline --interval 300 --limit 100
```

Single cycle mode:
```powershell
python -m backend.ingestion.run_realtime_pipeline --once --limit 100
```

### 5) Optional: force orchestration backfill
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/orchestration/run" -ContentType "application/json" -Body '{"run_llm": true, "limit": 200}'
```

## Validation Checklist

1. Health endpoint returns OK.
2. Ingestion pipeline logs show source polling and Kafka publish counts.
3. Cases populate with triage/deep-audit states.
4. Dashboard and inbox reflect new/updated cases.
5. Realtime timeline updates from websocket stream.

## Operational Notes

### Why coverage can appear tiny
If total transactions are high and audited subset is small, coverage can be mathematically very small (for example `<0.01%`).

### Why ROI can remain zero
ROI stays zero until action plans are completed and realized savings are recorded.

### Why some forensic fields show pending/not-applicable
- **Pending**: triage/deep-audit has not completed yet.
- **Not applicable**: case did not escalate, so deep-audit fields are intentionally absent.

## Current Constraints

- CPPP page structure can change over time; adapter includes robust HTML fallback but may require selector refreshes when the portal updates markup.
- Some OCDS registry resources are large or intermittently unavailable; adapter fetches JSON resources progressively and skips failing URLs.
- Realtime transport remains snapshot-style websocket updates, not full CDC/event sourcing.
- RBAC/SSO/approval controls are still future hardening tracks.

## Recommended Next Increments

1. Persist canonical source confidence and quality flags per record.
2. Add graph-risk scores directly into case prioritization.
3. Add per-source freshness SLAs and ingestion lag dashboards.
4. Add role-based workflow and approval gates for case closure.
