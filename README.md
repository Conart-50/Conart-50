## Bank Statement Budget Planner

This repository now includes a minimal Python workflow to:

- load **3–6 months** of bank statements (CSV/XLSX),
- clean and normalize transaction data,
- organize spending into categories,
- analyze monthly spending and top expense categories,
- generate a budget-cut plan for upcoming months.

### Main script

- `/home/runner/work/Conart-50/Conart-50/budget_planner.py`

### Required libraries

- `pandas`
- `numpy`
- `scikit-learn`

### Usage

```bash
python budget_planner.py statement_jan.csv statement_feb.csv statement_mar.csv --output-dir outputs
```

### Output files

- `outputs/monthly_spending.csv`
- `outputs/top_categories.csv`
- `outputs/budget_plan.csv`
- `outputs/summary.json`
