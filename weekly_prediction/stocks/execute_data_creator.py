from pathlib import Path
import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path when running as a script
_this_file = _Path(__file__).resolve()
_project_root = _this_file.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from stocks.stock_data_creator import StockDataCreator


def main():
    creator = StockDataCreator(config_path=Path("stocks/conf/config.ini"), data_dir=Path("data"))
    creator.export_tables(
        ["last_stock_earnings",  "earnings_estimates", "earnings_history", 
         "growth_estimates", "quarterly_income", "quarterly_revenue",
         "revenue_estimates","stocksinfp"],
        name_mapping={"last_stock_earnings": "stock_earnings", "stocksinfp" : "stock_prices"})


if __name__ == "__main__":
    main()