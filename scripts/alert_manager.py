"""Simple alerting utilities for Valvulin deployments."""
from __future__ import annotations

import asyncio
import logging
import os
from email.message import EmailMessage
from typing import Optional

import smtplib

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

LOGGER = logging.getLogger("valvulin.alerts")


async def send_telegram(message: str) -> None:
    token = os.getenv("VALVULIN_TELEGRAM_TOKEN")
    chat_id = os.getenv("VALVULIN_TELEGRAM_CHAT")
    if not token or not chat_id:
        LOGGER.debug("Telegram credentials not configured; skipping alert")
        return
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for Telegram alerts")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data={"chat_id": chat_id, "text": message}) as response:
            if response.status >= 400:
                LOGGER.error("Failed to send telegram alert: %s", await response.text())


def send_email(subject: str, body: str) -> None:
    sender = os.getenv("VALVULIN_ALERT_FROM")
    recipient = os.getenv("VALVULIN_ALERT_TO")
    smtp_server = os.getenv("VALVULIN_SMTP_SERVER")
    if not all([sender, recipient, smtp_server]):
        LOGGER.debug("Email credentials not configured; skipping alert")
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(body)
    with smtplib.SMTP(smtp_server) as client:
        client.send_message(msg)


async def notify_error(message: str, *, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    loop = loop or asyncio.get_event_loop()
    LOGGER.error("Engine error: %s", message)
    tasks = [loop.create_task(send_telegram(message))]
    await asyncio.gather(*tasks, return_exceptions=True)
    send_email("Valvulin alert", message)
