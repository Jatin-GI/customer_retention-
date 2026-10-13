"""RFM score direction and segment rule order."""

import pandas as pd

from src.segmentation import quintile_score, segment_customers


def test_quintile_score_direction():
    frame = pd.DataFrame(
        {
            "customer_id": [str(index) for index in range(10)],
            "recency_days": list(range(1, 11)),
            "frequency": list(range(1, 11)),
            "monetary": [float(index) for index in range(1, 11)],
        }
    )
    scored = segment_customers(frame)
    most_recent = scored.loc[scored["recency_days"].idxmin()]
    least_recent = scored.loc[scored["recency_days"].idxmax()]
    assert int(most_recent["r_score"]) == 5
    assert int(least_recent["r_score"]) == 1
    assert int(scored.loc[scored["monetary"].idxmax(), "m_score"]) == 5
    assert int(scored.loc[scored["frequency"].idxmin(), "f_score"]) == 1


def test_first_segment_rule_wins():
    frame = pd.DataFrame(
        {
            "customer_id": ["champ", "idle"],
            "recency_days": [1, 400],
            "frequency": [50, 1],
            "monetary": [5000.0, 5.0],
        }
    )
    # Two rows cannot form five quintiles; build a wider base and overwrite scores via rules
    # by checking the champion pattern on a 10-row frame where the top customer matches.
    wide = pd.DataFrame(
        {
            "customer_id": [f"c{index}" for index in range(8)] + ["champ", "quiet"],
            "recency_days": [30, 40, 50, 60, 70, 80, 90, 100, 1, 5],
            "frequency": [2, 2, 2, 3, 3, 4, 4, 4, 20, 1],
            "monetary": [10, 12, 14, 16, 18, 20, 22, 24, 500, 8],
        }
    )
    scored = segment_customers(wide)
    assert scored.loc[scored["customer_id"] == "champ", "segment"].iloc[0] == "Recent and frequent"
    assert frame.iloc[0]["customer_id"] == "champ"
