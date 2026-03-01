<p align="center">
  <img src="https://img.shields.io/badge/🧬_ORACLE-Viral_Evolution_Engine-blueviolet?style=for-the-badge&labelColor=0f0c29" alt="ORACLE"/>
</p>

<h1 align="center">🧬 ORACLE</h1>

<p align="center">
  <em>What if we could predict how a virus will mutate — before it actually does?</em>
</p>

<p align="center">
  <a href="#-getting-started"><img src="https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white"/></a>
  <a href="#-the-dashboard"><img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/></a>
  <a href="#-the-brains-4-ml-models"><img src="https://img.shields.io/badge/ML-4_Models-7B2FF7?style=flat-square"/></a>
  <a href="#-the-agents"><img src="https://img.shields.io/badge/Agents-5_AI_Workers-00D2FF?style=flat-square"/></a>
  <a href="#%EF%B8%8F-production-ready"><img src="https://img.shields.io/badge/DevOps-K8s_Docker_Terraform-326CE5?style=flat-square&logo=kubernetes&logoColor=white"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/></a>
</p>

---

## 💡 The Problem

Every few months, a new viral variant shows up — BA.5, XBB.1.5, JN.1 — and we're always playing catch-up. By the time we sequence it, study it, and update vaccines, the virus has already moved on.

**ORACLE flips that on its head.**

Instead of reacting to what happened, ORACLE uses AI to forecast what's *about* to happen. It analyzes incoming genomic sequences, predicts which mutations are coming next, figures out whether those mutations will dodge our immune system, and then *designs vaccine candidates* for variants that don't even exist yet.

All of this happens autonomously. You press one button. Five AI agents do the rest.

---

## 🎬 See It In Action

Here's the full dashboard walkthrough — pipeline execution, every tab, every chart, scrolled through top to bottom:

<p align="center">
  <img src="assets/oracle_demo.webp" alt="ORACLE Dashboard Demo — Full Walkthrough" width="100%"/>
</p>

> ☝️ *This shows the real pipeline running: 200 sequences analyzed → 537 mutations detected → 8 lineages scored → 10 vaccine candidates designed — all in under 1 second.*

---

## 🧠 How It Works

Think of ORACLE as a team of five AI specialists, each doing one job really well, passing their work to the next:

```
  You click "Run Pipeline"
         ↓
  📡 Surveillance Agent    — Scans incoming sequences. Spots new mutations. Raises alerts.
         ↓
  🔮 Evolution Agent       — Asks: "What mutations are coming next?" Uses transformers to predict.
         ↓
  🛡️ Escape Agent         — Asks: "Can these mutations dodge our antibodies?" Scores escape risk.
         ↓
  💉 Vaccine Agent         — Designs 10 vaccine candidates using 5 different strategies.
         ↓
  📋 Report Agent          — Writes a full WHO-style intelligence briefing. Ready to share.
```

The whole thing runs in under a second. No human intervention needed.

---

## 🧪 The Brains: 4 ML Models

These aren't toy models — they're based on real architectures used in computational biology:

| Model | What It Does | Why It Matters |
|-------|-------------|----------------|
| **Protein Language Model** | Uses ESM-2 (transformer) to create 1280-d embeddings of viral proteins | Understands the "language" of proteins — which mutations break things, which don't |
| **Evolution Forecaster** | Temporal transformer trained on historical mutation patterns | Predicts which mutations will appear in 30/60/90/120 days |
| **Immune Escape Scorer** | GNN that scores antibody evasion across 4 structural classes | Uses real deep mutational scanning data from 24 experimentally characterized mutations |
| **Vaccine Designer** | Multi-objective optimizer with 5 strategies | Designs antigens that target *future* variants, not just current ones |

The vaccine designer doesn't just pick one approach — it tries **consensus**, **mosaic**, **proactive**, **broadly-neutralizing**, and **stochastic optimization**, then ranks all candidates.

---

## 📊 The Dashboard

8 tabs, each telling part of the story:

| Tab | What You'll See |
|-----|----------------|
| 🏠 **Overview** | The big picture — metrics cards, executive summary, risk level, and active alerts |
| 📡 **Variant Tracker** | Which lineages are circulating, how many sequences per variant, mutation prevalence by region |
| 🔮 **Mutation Forecast** | Probability charts for predicted mutations, fitness vs. escape scatter plots |
| 🌳 **Phylogenetics** | Evolutionary landscape — how variants relate to each other, mutation counts per lineage |
| 🛡️ **Immune Escape** | Antibody escape scores broken down by 4 antibody classes, risk matrix (LOW → CRITICAL) |
| 💉 **Vaccine Candidates** | Radar chart comparing top 3 candidates, full ranked table with the #1 recommendation |
| 🤖 **Agent Monitor** | Pipeline execution status, per-agent timing, real-time progress tracking |
| 📈 **MLOps** | Drift detection, model retraining controls, experiment tracking, model registry |

Dark theme. Plotly charts. Interactive everything.

---

## 🔌 MCP Servers — 20 Tools

ORACLE exposes its brain through 4 **Model Context Protocol** servers. Any AI agent or external system can call these tools:

**Genomic Server** (port 8100) — Search sequences, get lineage info, mutation stats, compare variants

**Protein Server** (port 8101) — Predict structure, compute binding affinity, analyze epitopes, score mutations

**Epidemiology Server** (port 8102) — Outbreak data, case counts, vaccination rates, SIR forecasting

**Phylogenetics Server** (port 8103) — Build trees, trace lineages, detect recombination events

That's 5 tools per server, 20 tools total — all accessible via MCP.

---

## 🧬 Real Biology, Simulated Data

ORACLE simulates **8 real SARS-CoV-2 variants** with their actual defining mutations:

| Variant | Why It Mattered | Key Mutations |
|---------|----------------|---------------|
| **Alpha** (B.1.1.7) | First major variant of concern | N501Y enhanced ACE2 binding |
| **Beta** (B.1.351) | Showed immune escape was possible | E484K — the first big escape mutation |
| **Delta** (B.1.617.2) | Highest transmissibility at the time | L452R + P681R fuselage cleavage |
| **Omicron BA.1** | Rewrote the rules — 30+ mutations | Massive antibody escape |
| **BA.2, BA.5** | Kept evolving within Omicron | Progressive immune evasion |
| **XBB.1.5** ("Kraken") | Most immune-evasive to date | F486P — novel escape mechanism |
| **JN.1** | Latest dominant variant | L455S + ongoing convergent evolution |

The escape model uses **24 real mutations** from deep mutational scanning studies — the same data that papers in *Nature* and *Cell* use.

---

## ⚙️ Production-Ready

This isn't a notebook project. ORACLE has full DevOps infrastructure:

| Layer | What's There |
|-------|-------------|
| **Containers** | `Dockerfile` + `docker-compose.yml` with 8 services |
| **Kubernetes** | Deployment with HPA auto-scaling (2–10 pods), CronJob for scheduled pipeline runs with GPU nodes |
| **Terraform** | Full AWS infra — VPC, EKS cluster (general + GPU nodes), S3 buckets for data & models |
| **CI/CD** | GitHub Actions: lint → test → integration → Docker build |
| **Monitoring** | Prometheus + Grafana dashboards |
| **MLOps** | Experiment tracking, model registry, PSI drift detection, automated retraining |
| **Data Versioning** | DVC pipeline with reproducible stages |

---

## 🚀 Getting Started

```bash
# Clone it
git clone https://github.com/Nikhilchapkanade/Oracle.git
cd Oracle

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run dashboard/app.py
```

Then click **"🚀 Run ORACLE Pipeline"** in the sidebar. That's it.

Want the full stack with monitoring?

```bash
docker-compose --profile mcp --profile mlops --profile monitoring up
```

---

## 🗂️ Project Structure

```
oracle/
├── src/
│   ├── core/           → Data models (15+ Pydantic schemas) + SQLite database
│   ├── ingestion/      → Sequence ingestion, mutation analysis, phylogenetic trees
│   ├── ml/             → ESM-2 protein LM, evolution forecaster, GNN escape, vaccine designer
│   ├── mcp_servers/    → 4 MCP servers (20 tools total)
│   ├── agents/         → 5 specialized agents + orchestrator
│   └── mlops/          → Experiment tracking, drift detection, retraining
├── dashboard/          → Streamlit app (8 tabs, dark theme)
├── infra/              → K8s manifests, Terraform, Prometheus
├── .github/workflows/  → CI/CD pipeline
├── Dockerfile          → Container build
└── docker-compose.yml  → 8-service stack
```

---

## 📬 What ORACLE Outputs

When the pipeline finishes, you get a **WHO-style intelligence briefing** with:

- ✅ Total sequences analyzed and unique mutations detected
- 🚨 Risk assessment (LOW → MODERATE → HIGH → CRITICAL)
- 🔴 Active alerts (convergent evolution, ACE2 contact mutations, RBD hotspots)
- 🛡️ Immune escape scores for every lineage, broken down by 4 antibody classes
- 💉 Top 10 vaccine candidates with multi-objective scores
- 📋 5 actionable recommendations

Everything is structured, machine-readable, and ready for downstream consumption.

---

## 📄 License

MIT — use it however you want.

---

<p align="center">
  <strong>🧬 Built to predict what's next — not just what's now.</strong>
</p>
