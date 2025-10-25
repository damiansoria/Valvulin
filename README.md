# Valvulin

Valvulin is a modular algorithmic trading research environment. The project ships with
an opinionated Python package layout, structured logging, and configuration tooling to
kick-start quantitative strategy development.

## Project structure

```
valvulin/
├── core/          # configuration, logging, bootstrapping helpers
├── data/          # market data ingestion utilities
├── strategies/    # strategy base classes and signal definitions
├── risk/          # risk engine components
├── execution/     # order routing abstractions
└── analytics/     # performance analysis helpers
```

Configuration files live under `valvulin/config/` and include a YAML file for runtime
settings as well as an `.env` template for secret values.

## Getting started

Create a virtual environment and install the dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Next, copy the environment template and provide your API keys:

```bash
cp valvulin/config/.env.example valvulin/config/.env
```

Review and adjust `valvulin/config/config.yaml` to match your preferred risk
parameters and data directories.

Finally, bootstrap logging before running your application:

```python
from valvulin import ConfigLoader, setup_logging

setup_logging()
config = ConfigLoader().load()
```

You can now initialize the remaining subsystems (data feeds, strategies, execution,
and analytics) using the provided modules.
