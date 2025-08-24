from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class SimpleAgent:
    top_df: pd.DataFrame

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = inputs.get("input", "").strip()
        if not isinstance(self.top_df, pd.DataFrame) or self.top_df.empty:
            return {"output": "I don't have any results loaded yet. Click Find Top Stocks first."}

        # Basic helpers
        def to_float(x: Any) -> Optional[float]:
            try:
                return float(str(x).replace('%', ''))
            except Exception:
                return None

        df = self.top_df.copy()
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()

        # Try to find tickers mentioned in the prompt
        mentioned = []
        words = [w.strip().upper() for w in prompt.replace(',', ' ').split() if w.strip()]
        tickers = set(df['ticker'].unique().tolist()) if 'ticker' in df.columns else set()
        for w in words:
            if w in tickers:
                mentioned.append(w)

        # Build a concise response
        lines = []
        if mentioned:
            for t in mentioned[:5]:
                row = df[df['ticker'] == t].head(1)
                if row.empty:
                    continue
                r = row.iloc[0]
                pred = None
                for c in [
                    'predicted_return_pct',
                    'xgb_predicted_return_pct',
                    'lstm_predicted_return_pct',
                    'xgb_pred',
                    'lstm_pred',
                ]:
                    if c in row.columns:
                        pred = to_float(r.get(c))
                        if pred is not None:
                            break
                conf = None
                for c in [
                    'confidence_score',
                    'xgb_confidence_score',
                    'lstm_confidence_score',
                    'confidence_score_xgb',
                    'confidence_score_lstm',
                ]:
                    if c in row.columns:
                        val = to_float(r.get(c))
                        if val is not None:
                            conf = val if conf is None else max(conf, val)
                ens = to_float(r.get('ensemble_score')) if 'ensemble_score' in row.columns else None
                parts = [f"{t}:"]
                if pred is not None:
                    parts.append(f"pred ~ {pred:.2f}")
                if conf is not None:
                    parts.append(f"conf ~ {conf:.0f}")
                if ens is not None:
                    parts.append(f"ens ~ {ens:.2f}")
                lines.append(" ".join(parts))

        # Add a short top summary if no tickers or to complement
        if not lines:
            # Top 5 by ensemble/xgb/lstm as available
            order_cols = [
                'ensemble_score', 'xgb_score_adj', 'xgb_score',
                'xgb_predicted_return_pct', 'lstm_predicted_return_pct'
            ]
            sdf = df.copy()
            for c in order_cols:
                if c in sdf.columns:
                    sdf = sdf.sort_values(c, ascending=False)
                    break
            head = sdf.head(5)
            for _, r in head.iterrows():
                t = str(r.get('ticker', '')).upper()
                pred = None
                for c in [
                    'predicted_return_pct', 'xgb_predicted_return_pct',
                    'lstm_predicted_return_pct', 'xgb_pred', 'lstm_pred'
                ]:
                    if c in df.columns:
                        pred = to_float(r.get(c))
                        if pred is not None:
                            break
                ens = to_float(r.get('ensemble_score')) if 'ensemble_score' in df.columns else None
                parts = [t]
                if pred is not None:
                    parts.append(f"pred {pred:.2f}")
                if ens is not None:
                    parts.append(f"ens {ens:.2f}")
                lines.append(" - ".join(parts))

        if not lines:
            lines.append("I couldn't extract details from the current results. Try asking about a specific ticker (e.g., 'What about AAPL?').")

        return {"output": "\n".join(lines)}


def create_chatbot_agent(top_df: pd.DataFrame) -> SimpleAgent:
    """Create a lightweight local agent that answers questions from the loaded DataFrame.
    This avoids external API dependencies while providing useful summaries.
    """
    return SimpleAgent(top_df=top_df if isinstance(top_df, pd.DataFrame) else pd.DataFrame())


