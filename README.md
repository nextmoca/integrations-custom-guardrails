# tfy-custom-guardrails

TrueFoundry AI Gateway custom-guardrail integrations. One repo, many self-contained integrations.

Each `integrations/<vendor>/` directory is a standard FastAPI Docker container that conforms to the [TrueFoundry custom-guardrail HTTP contract](docs/gateway-contract.md). The TFY gateway calls these services at the `llm_input` and `llm_output` hooks; they return a verdict and the gateway honors it.

**Where to host the wrapper is up to you.** The wrapper is just a Docker container — run it on ECS, Cloud Run, Kubernetes, on-prem, a dev laptop, or anywhere else Docker runs. The only TFY-side requirement is that the resulting public HTTPS URL is reachable from the TFY Gateway and registered in the dashboard. Each integration in this repo includes a `deploy.py` that shows one example of hosting (a TrueFoundry Service deploy via the TrueFoundry Python SDK).

> **New integrator? Start here**: [`docs/add-a-new-integration.md`](docs/add-a-new-integration.md)
>
> **Working with Claude Code? Open [`CLAUDE.md`](CLAUDE.md) first.**

## Current integrations

| Integration | Vendor | Endpoints |
|---|---|---|
| [`integrations/nemo/`](integrations/nemo/) | NVIDIA NeMo Guardrails (LLM-judged rails) | `/self-check-input`, `/self-check-output` |
| [`integrations/guardrails-ai/`](integrations/guardrails-ai/) | Guardrails AI Hub validators (local heuristics) | `/detect-pii-{input,output}`, `/secrets-present-{input,output}`, `/toxic-language-{input,output}`, `/profanity-free-output` |
| [`integrations/lasso-security/`](integrations/lasso-security/) | [Lasso Security](https://server.lasso.security) API v3 (SaaS classify + classifix) | `/lasso-classify`, `/lasso-classify-output`, `/lasso-classifix`, `/lasso-classifix-output` |
| [`integrations/coreweave-weave/`](integrations/coreweave-weave/) | CoreWeave Weave scorers (Celadon toxicity classifier; local ML) | `/toxicity-input`, `/toxicity-output` |
| [`integrations/arthur-ai/`](integrations/arthur-ai/) | [Arthur GenAI Engine](https://platform.arthur.ai) stateless validation API (SaaS) | `/validate-input`, `/validate-output` |
| [`integrations/verra/`](integrations/verra/) | [Verra](https://helloverra.com) managed AI governance (SaaS validate + mutate) | `/scan-input`, `/redact-input`, `/scan-output`, `/redact-output` |
| [`integrations/hiddenlayer/`](integrations/hiddenlayer/) | [HiddenLayer AISec](https://hiddenlayer.com) Detection v2 API (validate + inline redact) | `/validate-input`, `/validate-output`, `/redact-input`, `/redact-output` |
| [`integrations/next-moca/`](integrations/next-moca/) | [Next Moca](https://nextmoca.com) context selection for tool results (SaaS mutate) | `/select-input` |

Each integration ships with its own `deploy.py` (a TrueFoundry Python SDK example) that prints the resolved public URL after a successful run. Use it as-is, swap it for your own deploy step (ECS task, Cloud Run service, Kubernetes manifest, etc.), or skip it entirely — the URL is what matters, not the hosting path.

Add a new integration: see [`docs/add-a-new-integration.md`](docs/add-a-new-integration.md).

## Repo layout

```
tfy-custom-guardrails/
├── README.md                       This file
├── CLAUDE.md                       Context for Claude Code sessions
├── docs/                           Cross-integration documentation
│   ├── gateway-contract.md         THE TFY custom-guardrail HTTP contract (single source of truth)
│   ├── integrations-overview.md    What custom guardrails are, why this repo exists, how integrations fit together
│   ├── guardrail-test-phrases.md   Vendor-agnostic test prompt catalog
│   └── add-a-new-integration.md    Step-by-step contributor flow
├── .claude/skills/                 Reusable skill for new integrations
│   └── truefoundry-custom-guardrail/
└── integrations/
    ├── _template/                  Skeleton — `cp -r _template/ <new-vendor>/`
    ├── nemo/                       NVIDIA NeMo Guardrails wrapper (example)
    ├── guardrails-ai/              Guardrails AI Hub validators wrapper (example)
    ├── lasso-security/             Lasso Security classify/classifix wrapper (SaaS)
    ├── coreweave-weave/            CoreWeave Weave scorers wrapper (Celadon toxicity)
    ├── arthur-ai/                  Arthur GenAI Engine validate API wrapper (SaaS)
    ├── verra/                      Verra managed AI governance wrapper (SaaS)
    ├── hiddenlayer/                HiddenLayer AISec Runtime Security wrapper (SaaS)
    └── next-moca/                  Next Moca context selection for tool results (SaaS)
```

## Design principle

**Each integration is fully self-contained.** No shared `entities.py`, no shared auth helpers, no shared deploy.py. The gateway contract is small enough that contract drift between integrations is handled at code-review time, not by enforced sharing. Read the rationale in [`docs/gateway-contract.md`](docs/gateway-contract.md) "Why no shared code".

Trade-offs we accept:

| Decision | Trade-off |
|---|---|
| Per-integration `entities.py` | When the gateway contract changes, update each integration's copy. Coordinated PR. |
| Per-integration `main.py`, `deploy.py`, `Dockerfile` | Each integration deploys independently with full freedom to pin different dependencies. |
| Per-integration venv | Each integration has its own `.venv/`. Vendor deps cannot conflict across integrations. |
| Cross-integration docs at top level | Single place for the contract, the demo prompts, the contributor flow, the skill. |

## Quick start

Clone, pick an integration to work on, work inside its directory:

```bash
git clone <repo-url>
cd tfy-custom-guardrails/integrations/<vendor>
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env  # fill in the values
.venv/bin/uvicorn main:app --reload --port 8000
```

To add a new vendor:

```bash
cp -r integrations/_template integrations/<new-vendor>
# Edit integrations/<new-vendor>/{main.py, guardrail/, requirements.txt, deploy.py, README.md, ...}
# Run tests, host the wrapper (anywhere — see the per-integration READMEs for examples),
# register in the TFY Gateway dashboard.
# Full step-by-step: docs/add-a-new-integration.md
```

## Where the wider context lives

- **What custom guardrails are + how to think about them**: [`docs/integrations-overview.md`](docs/integrations-overview.md).
- **The HTTP contract**: [`docs/gateway-contract.md`](docs/gateway-contract.md). Single source of truth.
- **Vendor-agnostic demo prompts** (allow controls + violations across PII / secrets / toxicity / jailbreak / etc.): [`docs/guardrail-test-phrases.md`](docs/guardrail-test-phrases.md).
- **Reusable skill** that future Claude Code sessions or contributors can use to scaffold a new integration: [`.claude/skills/truefoundry-custom-guardrail/`](.claude/skills/truefoundry-custom-guardrail/).
