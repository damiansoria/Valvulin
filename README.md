# Valvulin Trading Engine

Valvulin es un bot de trading modular orientado a operar múltiples símbolos y marcos temporales en paralelo. El motor está diseñado para ejecutarse 24/7 con configuraciones declarativas, estrategias dinámicas y un enfoque asincrónico para consumir datos de mercado vía websockets o feeds personalizados.

## Características principales

- **Motor central** (`core/engine.py`) que inicializa _data feeds_, estrategias, control de riesgo y ejecución de órdenes.
- **Configuraciones en YAML/JSON** con soporte para habilitar o deshabilitar estrategias por símbolo y plazo.
- **Overrides por CLI** (`--enable-strategy` / `--disable-strategy`) para pruebas rápidas sin editar archivos.
- **Scheduler asíncrono** capaz de disparar evaluaciones periódicas por timeframe (1m, 5m, 1h, etc.) manteniendo compatibilidad con feeds en tiempo real.
- **Integración de alertas** (Telegram/Email) mediante utilidades en `scripts/alert_manager.py` para monitoreo continuo.
- **Paquetes de despliegue** listos para Docker, systemd y supervisord.

## Estructura del proyecto

```
core/
  config.py          # Lectura de YAML/JSON y CLI overrides
  data_feed.py       # Gestión de feeds y abstracciones base
  engine.py          # Bucle principal de orquestación
  execution.py       # Manejo de órdenes
  risk.py            # Controles de riesgo simples
  scheduler.py       # Scheduler asíncrono recurrente
  strategy.py        # Registro y ciclo de vida de estrategias
strategies/
  basic.py           # Estrategia de ejemplo
scripts/
  start_bot.sh       # Script de arranque simple
  alert_manager.py   # Herramientas para alertas Telegram/Email
config.example.yaml  # Configuración de referencia
requirements.txt     # Dependencias necesarias
```

## Configuración

1. Copia `config.example.yaml` a `config.yaml` y ajusta parámetros.
2. Define los _handlers_ de feeds y estrategias utilizando rutas tipo `paquete.modulo:Clase`.
3. Para habilitar/deshabilitar estrategias por símbolo/plazo utiliza el campo `enabled` o sobreescribe desde CLI:

```bash
python -m core.engine --config config.yaml \
    --enable-strategy BTCUSDT:1m:echo-btc \
    --disable-strategy ETHUSDT:1m:echo-eth
```

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m core.engine --config config.yaml
```

Durante la ejecución el motor escuchará señales SIGINT/SIGTERM para apagado seguro. El scheduler ejecutará los intervalos definidos mientras los feeds consumen datos asincrónicamente.

## Monitoreo y alertas

`scripts/alert_manager.py` proporciona utilidades para notificar errores críticos:

- Exporta `VALVULIN_TELEGRAM_TOKEN` y `VALVULIN_TELEGRAM_CHAT` para alertas vía Telegram.
- Exporta `VALVULIN_ALERT_FROM`, `VALVULIN_ALERT_TO` y `VALVULIN_SMTP_SERVER` para alertas por correo.

Integra `notify_error` en tus estrategias o envoltorios de ejecución para disparar alertas cuando ocurra una excepción crítica.

## Despliegue 24/7

### Docker

1. Ajusta `config.yaml` y coloca el archivo junto al repositorio.
2. Construye la imagen:

   ```bash
   docker compose -f deploy/docker-compose.yml build
   ```

3. Lanza el contenedor:

   ```bash
   VALVULIN_TELEGRAM_TOKEN=xxx VALVULIN_TELEGRAM_CHAT=yyy \
   VALVULIN_ALERT_FROM=bot@example.com VALVULIN_ALERT_TO=ops@example.com \
   VALVULIN_SMTP_SERVER=smtp.example.com \
   docker compose -f deploy/docker-compose.yml up -d
   ```

### systemd

1. Copia el repositorio a `/opt/valvulin`.
2. Copia `deploy/systemd/trading-bot.service` a `/etc/systemd/system/`.
3. (Opcional) Crea `/etc/valvulin/bot.env` con las variables de entorno.
4. Recarga e inicia:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now trading-bot.service
   ```

### supervisord

1. Copia `deploy/supervisord.conf` a `/etc/supervisor/conf.d/valvulin.conf`.
2. Asegúrate de que `/opt/valvulin` contenga el código y `config.yaml`.
3. Reinicia supervisord: `sudo supervisorctl reread && sudo supervisorctl update`.

## Buenas prácticas de operación

- Configura _logging_ para almacenar en disco o sistemas centralizados (p. ej. `journald`, CloudWatch).
- Utiliza `scripts/start_bot.sh` en entornos donde se requiera lanzamiento manual.
- Monitorea métricas clave (latencias del scheduler, estado de los feeds, órdenes ejecutadas) mediante herramientas externas o dashboards personalizados.

## Desarrollo

- Ejecuta pruebas manuales con el feed `MockDataFeed`.
- Añade estrategias en el directorio `strategies/` implementando `BaseStrategy`.
- Extiende `RiskManager` y `ExecutionEngine` según tus necesidades de cumplimiento y conectores de exchange.

## Licencia

Este proyecto se entrega tal cual. Ajusta la licencia según los requerimientos de tu organización.
