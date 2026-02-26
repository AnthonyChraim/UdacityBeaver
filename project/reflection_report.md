# Beaver's Choice Paper Company - Multi-Agent System Report

## 1: Architecture Summary

The implementation uses 4 agents:

1. **OrchestratorAgent**

- Entry point for each request.
- Parses date + items, validates request shape, routes work to worker agents.
- Returns customer-facing responses with fulfillment rationale.

2. **InventoryAgent**

- Checks current stock and reorder needs per item.
- Uses supplier lead-time logic and cash checks before approving reorder.
- Creates stock order transactions when needed.

3. **QuotingAgent**

- Calculates subtotal from catalog unit prices.
- Uses historical quote records and order size to apply discounts.
- Produces explainable quote text.

4. **SalesAgent**

- Records sales transactions by line item.
- Fetches updated financial report after sale.

Framework integration:

- The system includes a `smolagents` delegation layer (`SmolagentsDelegator`) that defines framework worker agents (`CodeAgent`) for inventory, quoting, and sales.
- When `UDACITY_OPENAI_API_KEY` is present, the orchestrator delegates worker decisions through these framework agents.
- If no key is present, the same flow runs through deterministic local workers for reproducible offline testing.

## 2: Tool Mapping to Starter Helper Functions

All required starter helper functions are wrapped in tool definitions:

- `inventory_snapshot_tool` -> `get_all_inventory`
- `stock_level_tool` -> `get_stock_level`
- `supplier_eta_tool` -> `get_supplier_delivery_date`
- `cash_balance_tool` -> `get_cash_balance`
- `quote_history_tool` -> `search_quote_history`
- `create_transaction_tool` -> `create_transaction`
- `financial_report_tool` -> `generate_financial_report`

## 3: Brief Function Descriptions (Starter Code Review)

- `generate_sample_inventory`: creates a reproducible random subset of paper items with initial stock/min levels.
- `init_database`: initializes SQLite tables (`transactions`, `quote_requests`, `quotes`, `inventory`) and seeds starting data.
- `create_transaction`: writes one stock order or sales transaction.
- `get_all_inventory`: computes item stock snapshot as-of a date from transaction history.
- `get_stock_level`: computes stock for one item as-of a date.
- `get_supplier_delivery_date`: estimates supplier lead time based on order quantity.
- `get_cash_balance`: computes sales minus stock purchase costs as-of a date.
- `generate_financial_report`: summarizes cash, inventory value, assets, and top sellers.
- `search_quote_history`: returns past quotes that match request/search terms.

## 4: Evaluation Results

The system was run using `quote_requests_sample.csv` via `run_test_scenarios()`.

- Output file generated: `test_results.csv`
- Total requests evaluated: **20**
- Requests fulfilled (response contains "Quote approved"): **10**
- Requests with cash-balance change: **9**
- Not all requests were fulfilled: **True**

Observed unfulfilled reasons include:

- out-of-catalog items (for example: balloons, A3-only items, cardboard)
- supplier lead-time misses vs. due date (for example: earliest delivery after required date)

This satisfies rubric expectations that:

- at least three requests change cash balance,
- at least three quote requests are fulfilled,
- not all requests are fulfilled, with explicit reasons.

## 5: Strengths

- Clear orchestration and role separation.
- Transparent customer-facing outcomes with concrete reasons.
- Database-backed decisions for stock, cash, and transactions.
- Deterministic behavior suitable for repeatable grading and debugging.

## 6: Areas for Improvement and Next Steps

1. Improve NLP item extraction:

- Current parser is regex-based and can still over-capture phrase fragments in some cases.
- Replace with structured extraction (LLM function-calling or tighter grammar parser) for cleaner line-item detection.

2. Add negotiation and partial-fulfillment options:

- Today, unknown item mentions can block full fulfillment.
- A stronger policy would propose closest alternatives and optional partial shipments.

3. Add business advisor analytics (optional stretch):

- Analyze historical transactions to recommend reorder thresholds, discount tuning, and seasonal inventory adjustments.
