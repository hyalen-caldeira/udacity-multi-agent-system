# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent system for Munder Difflin Paper Company (Udacity project). Automates business operations (inventory, quoting, sales) using a 4-agent orchestration pattern with OpenAI GPT-4o via the `smolagents` framework.

## Setup & Running

```bash
source venv/bin/activate
pip install -r requirements.txt
pip install smolagents

# Run the system (processes test scenarios from quote_requests_sample.csv)
python project_starter.py
```

Requires `UDACITY_OPENAI_API_KEY` in `.env` file. Uses custom OpenAI-compatible proxy via `OpenAIServerModel` (endpoint: `https://openai.vocareum.com/v1`).

## Architecture

All code lives in a single file: `project_starter.py`.

### Agent Hierarchy (Orchestrator Pattern)

```
manager_agent (orchestrator, no tools, max_steps=18)
├── inventory_agent (check_inventory, reorder_stock, max_steps=12)
├── quoting_agent (search_quotes, lookup_item_price, max_steps=12)
└── sales_agent (get_financial_report, check_delivery, process_sale, max_steps=12)
```

### Processing Pipeline

For each customer request: **Inventory check/reorder → Price quote with bulk discounts → Sale processing**

### Key Sections in project_starter.py

- **Lines 17-71**: Product catalog (`paper_supplies` — 45 items across paper, product, large_format, specialty)
- **Lines 75-583**: Helper functions (inventory, transactions, financial reports, quote history search)
- **Lines 593-605**: Model setup and catalog string construction
- **Lines 609-809**: `@tool` decorated functions callable by agents (6 tools total)
- **Lines 813-887**: Agent definitions (inventory, quoting, sales, manager)
- **Lines 892-981**: `run_test_scenarios()` — main entry point

### Data Layer

- **SQLite**: `munder_difflin.db` — **recreated from scratch on every run** via `init_database()`
- **Global engine**: `db_engine` (module-level) used by all helpers and tools
- **Tables**: `inventory`, `transactions`, `quotes`, `quote_requests`
- **Initial state**: $50,000 cash, ~40% catalog coverage (18 of 45 items), 200-800 units per stocked item
- **Input CSVs**: `quote_requests.csv`, `quotes.csv`, `quote_requests_sample.csv` (test scenarios)
- **Output**: `test_results.csv`

### Bulk Discount Tiers

- 100-499 units: 5%
- 500-999 units: 10%
- 1000+ units: 15%

### Delivery Lead Times (by quantity)

- ≤10 units: same day
- 11-100: +1 day
- 101-1000: +4 days
- >1000: +7 days

## Key Design Constraints

- Maximum 5 agents, text-based communication only
- Agents have isolated tool access (no cross-domain side effects)
- Tool functions use fuzzy/substring matching to map customer item names to exact catalog names
- System gracefully handles partial fulfillment when items are unavailable or not in catalog
- Stock levels are computed from transaction history (sum of stock_orders minus sales), not stored directly
