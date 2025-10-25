# Valvulin

Herramientas de analítica para registrar y evaluar el desempeño de un bot de trading.

## Componentes

- `analytics/trade_logger.py`: manejo de trades cerrados y posiciones abiertas mediante CSV.
- `analytics/performance.py`: cálculo de métricas agregadas (win rate, profit factor, max drawdown, expectancy).
- `analytics/plots.py`: generación de curvas de equity, histogramas de R y patrones por estrategia con Matplotlib.
- `analytics/dashboard.py`: dashboard de línea de comandos que muestra estado general, operaciones abiertas y estadísticas recientes.

## Uso rápido

```bash
# Registrar un trade
python -c "from analytics.trade_logger import TradeLogger; TradeLogger().log_trade('breakout', 1.5, 0.95, 1.2, 'Trade de ejemplo')"

# Consultar métricas
python -c "from analytics.trade_logger import TradeLogger; from analytics.performance import compute_performance_metrics; logger = TradeLogger(); metrics = compute_performance_metrics(logger.load_trades()); print(metrics)"

# Mostrar dashboard en CLI
python -m analytics.dashboard
```

Los archivos CSV se guardan en `data/` por defecto.
