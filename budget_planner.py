from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

COLUMN_ALIASES = {
    "date": {"date", "transaction date", "posted date", "value date"},
    "description": {"description", "details", "merchant", "narration", "memo"},
    "amount": {"amount", "debit", "withdrawal", "spent", "value"},
    "credit": {"credit", "deposit"},
}

CATEGORY_KEYWORDS = {
    "Housing": ["rent", "mortgage", "utility", "electric", "water", "internet"],
    "Food": ["grocery", "supermarket", "restaurant", "cafe", "food", "ubereats"],
    "Transport": ["uber", "lyft", "fuel", "gas", "train", "bus", "taxi"],
    "Shopping": ["amazon", "store", "mall", "clothing", "shop"],
    "Health": ["pharmacy", "hospital", "clinic", "medical", "doctor"],
    "Entertainment": ["netflix", "spotify", "cinema", "movie", "game"],
    "Fees": ["fee", "charge", "interest", "penalty"],
}


def _find_column(columns: Iterable[str], candidates: set[str]) -> str | None:
    lookup = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def load_statements(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        else:
            frame = pd.read_csv(path)
        frames.append(frame)

    if not frames:
        raise ValueError("No statement files were provided")

    df = pd.concat(frames, ignore_index=True)

    date_col = _find_column(df.columns, COLUMN_ALIASES["date"])
    desc_col = _find_column(df.columns, COLUMN_ALIASES["description"])
    amount_col = _find_column(df.columns, COLUMN_ALIASES["amount"])
    credit_col = _find_column(df.columns, COLUMN_ALIASES["credit"])

    if date_col is None or desc_col is None:
        raise ValueError("Could not detect date/description columns")

    if amount_col is None and credit_col is None:
        raise ValueError("Could not detect amount columns")

    cleaned = pd.DataFrame()
    cleaned["date"] = pd.to_datetime(df[date_col], errors="coerce")
    cleaned["description"] = df[desc_col].fillna("").astype(str).str.strip()

    if amount_col is not None:
        amount_series = pd.to_numeric(df[amount_col], errors="coerce")
        if amount_series.mean(skipna=True) < 0:
            amount_series = -amount_series
    else:
        amount_series = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    if credit_col is not None:
        credit_series = pd.to_numeric(df[credit_col], errors="coerce").fillna(0)
        amount_series = amount_series.fillna(0) - credit_series

    cleaned["amount"] = amount_series.fillna(0)
    cleaned = cleaned.dropna(subset=["date"]).copy()
    cleaned = cleaned[cleaned["amount"] > 0].copy()
    cleaned["month"] = cleaned["date"].dt.to_period("M").astype(str)

    return cleaned


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    def classify(description: str) -> str:
        lower_text = description.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in lower_text for keyword in keywords):
                return category
        return "Other"

    categorized = df.copy()
    categorized["category"] = categorized["description"].map(classify)
    return categorized


def monthly_spending(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["month", "category"], as_index=False)["amount"]
        .sum()
        .sort_values(["month", "amount"], ascending=[True, False])
    )
    return summary


def top_spending_categories(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    return (
        df.groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
        .head(top_n)
    )


def budget_cut_plan(monthly_summary: pd.DataFrame, cut_floor: float = 0.10) -> pd.DataFrame:
    if monthly_summary.empty:
        return pd.DataFrame(columns=["category", "projected_next_month", "recommended_budget"])

    data = monthly_summary.copy()
    data["month_index"] = pd.to_datetime(data["month"] + "-01", errors="coerce")

    recommendations = []
    category_rank = (
        monthly_summary.groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
        .reset_index(drop=True)
    )
    rank_lookup = {row.category: i for i, row in category_rank.iterrows()}

    for category, group in data.groupby("category"):
        group = group.sort_values("month_index")
        if group["month_index"].isna().all():
            continue

        x = np.arange(len(group), dtype=float).reshape(-1, 1)
        y = group["amount"].to_numpy(dtype=float)

        if len(group) >= 2:
            model = LinearRegression().fit(x, y)
            projected = float(model.predict(np.array([[len(group)]], dtype=float))[0])
        else:
            projected = float(y.mean())

        projected = max(projected, 0.0)

        rank = rank_lookup.get(category, 99)
        cut_rate = cut_floor + (0.10 if rank == 0 else 0.05 if rank < 3 else 0.0)
        recommended = projected * (1 - cut_rate)

        recommendations.append(
            {
                "category": category,
                "projected_next_month": round(projected, 2),
                "recommended_budget": round(max(recommended, 0.0), 2),
                "cut_rate": round(cut_rate, 2),
            }
        )

    return pd.DataFrame(recommendations).sort_values("projected_next_month", ascending=False)


def analyze(paths: list[Path], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned = load_statements(paths)
    categorized = categorize_transactions(cleaned)

    monthly = monthly_spending(categorized)
    top_categories = top_spending_categories(categorized)
    plan = budget_cut_plan(monthly)

    monthly_path = output_dir / "monthly_spending.csv"
    top_path = output_dir / "top_categories.csv"
    budget_path = output_dir / "budget_plan.csv"

    monthly.to_csv(monthly_path, index=False)
    top_categories.to_csv(top_path, index=False)
    plan.to_csv(budget_path, index=False)

    summary = {
        "transactions": str(len(categorized)),
        "months_covered": str(categorized["month"].nunique()),
        "monthly_spending": str(monthly_path),
        "top_categories": str(top_path),
        "budget_plan": str(budget_path),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze bank statements and build a category-based budget cut plan."
    )
    parser.add_argument(
        "statements",
        nargs="+",
        type=Path,
        help="Path(s) to CSV/XLSX statement files (recommended 3-6 months).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(args.statements, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
