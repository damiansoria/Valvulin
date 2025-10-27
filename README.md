# Valvulin Trading Engine

Valvulin es un bot de trading modular pensado para operar múltiples símbolos y marcos temporales en paralelo. El proyecto integra backtesting histórico, ejecución en vivo y herramientas de análisis para medir el desempeño de las estrategias.

## Descripción general

- **Motor central (`core/`)**. Coordina la configuración, el scheduler asíncrono, los _data feeds_, las estrategias y el motor de ejecución.
- **Ejecución flexible (`execution/`)**. Incluye un backtester histórico y conectores para brokers/exchanges. `core/execution.py` actúa como fachada interna mientras que `execution/live_executor.py` implementa la lógica específica de mercado.
- **Estrategias modulares (`strategies/`)**. Cada estrategia opera sobre `pandas.DataFrame` y entrega señales normalizadas en la columna `signal` (`1` compra, `-1` venta, `0` neutro).
- **Analytics (`analytics/`)**. Métricas de performance, gráficas interactivas con Plotly y un dashboard opcional para revisar resultados.

## Dependencias clave

| Paquete            | Uso principal                                  |
|--------------------|------------------------------------------------|
| `pandas`           | Manipulación de series de precios y OHLCV.     |
| `numpy`            | Cálculo numérico y métricas.                   |
| `plotly`           | Visualizaciones interactivas de backtests.     |
| `matplotlib`       | Gráficos estáticos de métricas.                |
| `aiohttp`, `websockets` | Consumo de datos en tiempo real.        |
| `python-binance` *(opcional)* | Conector para trading en vivo.    |

Instala todo con:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración rápida

1. Copia `config.example.yaml` a `config.yaml`.
2. Ajusta las secciones `data_feeds`, `strategies`, `risk` y `execution`. Cada estrategia se referencia con la ruta `paquete.modulo:Clase` y permite habilitarse por símbolo/timeframe (`enabled: true/false`).
3. (Opcional) Utiliza overrides por CLI para activar/desactivar estrategias sin editar el archivo:

```bash
python -m core.engine --config config.yaml \
    --enable-strategy BTCUSDT:1h:ema-macd \
    --disable-strategy ETHUSDT:15m:rsi-divergence
```

## Descarga de datos históricos desde Binance

- El feed `BinancePublicDataFeed` permite descargar velas OHLCV directamente desde el endpoint público de Binance **sin API key**.
- Respeta los límites públicos de Binance (máximo de 1200 requests por minuto); el cliente incluye pausas y reintentos exponenciales automáticos.

### Script de utilidad

```bash
python scripts/download_binance_data.py
```

Esto creará (o actualizará) `data/history/BTCUSDT_1h.csv` con datos desde el 1 de enero de 2024.

### Uso en `config.yaml`

```yaml
data_feeds:
  - name: binance-public
    type: binance-public
    params:
      data_dir: data/history
      cache: true
```

Cuando se utilice este feed desde el `engine`, se inicializará automáticamente sin credenciales adicionales:

```bash
python -m core.engine --config config.yaml
```

## Ejecución en vivo

```bash
python -m core.engine --config config.yaml
```

El motor inicializará los feeds, las estrategias y el scheduler. Se puede detener de forma segura con `Ctrl+C` (SIGINT) o enviando SIGTERM.

## Cómo correr un backtest paso a paso

1. **Preparar datos**: obtén un `DataFrame` OHLCV con índice `DatetimeIndex`.
2. **Generar señales**: instancia una estrategia y produce la columna `signal`.
3. **Ejecutar el backtester**: usa `execution.Backtester` para simular operaciones.
4. **Analizar métricas**: calcula métricas con `analytics.performance`.
5. **Visualizar resultados**: genera un gráfico interactivo con `analytics.signals_plot`.

Ejemplo completo:

```python
import pandas as pd
from execution import Backtester
from strategies import EMAMACDStrategy
from analytics.performance import compute_performance_metrics
from analytics.signals_plot import plot_backtest_signals

# 1) Datos de ejemplo
prices = pd.read_csv("data/BTCUSDT_1h.csv", parse_dates=["timestamp"], index_col="timestamp")

# 2) Generación de señales normalizadas
strategy = EMAMACDStrategy(ema_short=12, ema_long=26, macd_signal=9)
signals = strategy.generate_signals(prices)

# 3) Construir TradeSignal iterable y ejecutar el backtest
from execution.types import TradeSignal
trade_signals = [
    TradeSignal(
        symbol="BTCUSDT",
        timestamp=index.to_pydatetime(),
        side="BUY" if row.signal == 1 else "SELL",
        quantity=1.0,
        stop_loss=None,
    )
    for index, row in signals.iterrows() if row.signal != 0
]
backtester = Backtester(prices)
result = backtester.run(trade_signals)

# 4) Métricas de performance (Win rate, Profit factor, Expectancy, etc.)
metrics = compute_performance_metrics(result.trades)
print(metrics.to_dict())

# 5) Visualización interactiva
plot_backtest_signals(prices, result.trades, title="Backtest BTCUSDT")
```

Los archivos HTML de las gráficas se guardan en `analytics/plots_output/`.

## Agregar una nueva estrategia (`strategies/`)

1. Crea un archivo en `strategies/` que herede de `BaseStrategy`.
2. Implementa `generate_signals(self, data: pd.DataFrame) -> pd.DataFrame` y devuelve la columna `signal` con valores `1`, `0` o `-1`.
3. (Opcional) Expón información adicional en columnas extra (por ejemplo `inside_bar`).
4. Registra la clase en `strategies/__init__.py` si quieres importarla directamente.
5. Actualiza o agrega pruebas unitarias en `tests/test_strategies.py`.

### Plantilla mínima

```python
import pandas as pd
from strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.sort_index().copy()
        signals = pd.Series(0, index=frame.index, dtype=int)
        # Lógica de trading...
        return pd.DataFrame({"signal": signals})
```

## Visualización y analytics

- `analytics/performance.py`: calcula Win rate, Profit factor, Expectancy, Average R multiple y Max drawdown.
- `analytics/signals_plot.py`: crea gráficos Plotly con velas, señales de entrada/salida e indicadores opcionales.
- `analytics/plots.py`: utilidades Matplotlib (equity curve, distribución de R, patrones por estrategia).
- `analytics/dashboard.py`: interfaz (Streamlit/Panel) que puede abrir gráficos generados y métricas.

Para generar un gráfico de señales existente:

```python
from analytics.plots import plot_trade_signals
from analytics.signals_plot import plot_backtest_signals
# ...
plot_backtest_signals(ohlc_df, backtest_result.trades, indicators={"EMA 20": ohlc_df["ema20"]})
```

## Despliegue 24/7

Valvulin incluye configuraciones de referencia para Docker, systemd y supervisord en la carpeta `deploy/`. Ajusta `config.yaml`, define tus variables de entorno y utiliza los archivos provistos (`docker-compose.yml`, `trading-bot.service`, `supervisord.conf`).

## Buenas prácticas de operación

- Configura logs centralizados (journald, CloudWatch, Loki, etc.).
- Define límites de riesgo en `core/risk.py` y monitorea el resultado con `analytics.performance`.
- Automatiza la generación de reportes post-backtest con `analytics.signals_plot` y `analytics.performance`.
- Integra `scripts/alert_manager.py` para recibir notificaciones por Telegram o correo.

## Licencia

Este proyecto se entrega tal cual. Ajusta la licencia según los requerimientos de tu organización.
