# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent system for Munder Difflin Paper Company (Udacity project). Automates business operations (inventory, quoting, sales) using a 4-agent orchestration pattern with OpenAI GPT-4o via the `smolagents` framework.

## Setup & Running

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install smolagents

# Requires UDACITY_OPENAI_API_KEY in .env file
# Uses custom proxy: https://openai.vocareum.com/v1

# Run the system (processes test scenarios)
python project_starter.py
```

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

- **Lines 16-71**: Product catalog (45 items across Paper, Products, Large-format, Specialty)
- **Lines 74-583**: Helper functions (inventory, transactions, financial reports)
- **Lines 609-809**: `@tool` decorated functions callable by agents
- **Lines 813-869**: Agent definitions
- **Lines 874-958**: `run_test_scenarios()` — main entry point

### Data Layer

- **SQLite database**: `munder_difflin.db` (recreated on each run via `init_database()`)
- **Tables**: `inventory`, `transactions`, `quotes`, `quote_requests`
- **Initial state**: $50,000 cash, ~40% catalog coverage, 200-800 units per stocked item
- **Input data**: `quote_requests.csv`, `quotes.csv`, `quote_requests_sample.csv`
- **Output**: `test_results.csv`

### Bulk Discount Tiers

- 100-499 units: 5%
- 500-999 units: 10%
- 1000+ units: 15%

## Key Design Constraints

- Maximum 5 agents, text-based communication only
- Agents have isolated tool access (no cross-domain side effects)
- Fuzzy matching maps customer item names to exact catalog names
- System gracefully handles partial fulfillment when items are unavailable
