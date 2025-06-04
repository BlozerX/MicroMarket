import math
from src.config import Config, State, Trade
from src.backtest import MarketMaker

cfg = Config(
    fixed_spread      = 0.02,
    max_inventory     = 100,
    inventory_penalty = 0.001,
    fee_per_share     = 0.0002,
    slip_ppv          = 0.00005,
    fill_prob         = 0.9
)

def _test_quote_spread():
    # Ensure that our quoted ask–bid distance equals the configured spread
    q = MarketMaker(cfg)._quote_prices(100)
    assert math.isclose(q[1] - q[0], cfg.fixed_spread), "Spread mismatch"

def _test_inv_penalty():
    # Check that positive inventory skews bids down
    mm1 = MarketMaker(cfg)
    mm2 = MarketMaker(cfg)
    mm2.state.inventory = 10
    bid1 = mm1._quote_prices(100)[0]
    bid2 = mm2._quote_prices(100)[0]
    assert bid2 < bid1, "Inventory penalty not applied correctly"

# Run tests
_test_quote_spread()
_test_inv_penalty()
print("Unit-tests passed")
