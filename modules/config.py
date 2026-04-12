import yaml
import os
from box import ConfigBox

# Pfad zur config.yaml (eine Ebene höher als dieser Ordner)
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

with open(config_path, "r") as f:
    # ConfigBox erlaubt den Zugriff via cfg.filters.min_market_cap statt ['filters']['min_market_cap']
    cfg = ConfigBox(yaml.safe_load(f))
