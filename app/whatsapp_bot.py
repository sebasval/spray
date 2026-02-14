"""
WhatsApp Bot module for Spray Analyzer.
Handles incoming WhatsApp messages via Meta Cloud API webhook.
"""

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import JSONResponse
import httpx
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["WhatsApp Bot"])

# Environment variables
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "spray-verify-2025")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")

# Internal API config (calls itself to analyze)
SPRAY_API_URL = os.getenv("SPRAY_API_URL", "https://spray-production.up.railway.app")
SPRAY_API_EMAIL = os.getenv("SPRAY_API_EMAIL", "admin@almagricola.com")
SPRAY_API_PASSWORD = os.getenv("SPRAY_API_PASSWORD", "")


# --- Spray Analyzer API helpers ---

async def get_spray_token() -> str:
    """Get JWT token from Spray Analyzer API."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SPRAY_API_URL}/token",
            data={"username": SPRAY_API_EMAIL, "password": SPRAY_API_PASSWORD},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]


async def analyze_image_via_api(image_bytes: bytes, filename: str = "image.jpg") -> dict:
    """Send image to Spray Analyzer API and return results."""
    token = await get_spray_token()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SPRAY_API_URL}/analyze",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (filename, image_bytes, "image/jpeg")},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


# --- WhatsApp API helpers ---

async def download_whatsapp_media(media_id: str) -> bytes:
    """Download media from WhatsApp Cloud API."""
    async with httpx.AsyncClient() as client:
        # Step 1: Get media URL
        resp = await client.get(
            f"https://graph.facebook.com/v22.0/{media_id}",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            timeout=15,
        )
        logger.info(f"Media URL response: {resp.status_code} {resp.text[:200]}")
        resp.raise_for_status()
        media_url = resp.json()["url"]

        # Step 2: Download the media
        resp = await client.get(
            media_url,
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            timeout=30,
        )
        logger.info(f"Media download response: {resp.status_code}, size: {len(resp.content)} bytes")
        resp.raise_for_status()
        return resp.content


async def send_whatsapp_message(to: str, text: str):
    """Send a text message via WhatsApp Cloud API."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages",
            headers={
                "Authorization": f"Bearer {WHATSAPP_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": text},
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()


async def upload_whatsapp_media(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Upload media to WhatsApp Cloud API and return media ID."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/media",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            data={"messaging_product": "whatsapp", "type": mime_type},
            files={"file": ("analysis.png", image_bytes, mime_type)},
            timeout=30,
        )
        logger.info(f"Media upload response: {resp.status_code} {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["id"]


async def send_whatsapp_image(to: str, media_id: str, caption: str = ""):
    """Send an image message via WhatsApp Cloud API."""
    async with httpx.AsyncClient() as client:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "image",
            "image": {"id": media_id},
        }
        if caption:
            payload["image"]["caption"] = caption
        resp = await client.post(
            f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages",
            headers={
                "Authorization": f"Bearer {WHATSAPP_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()


# --- Message handlers ---

async def handle_text_message(sender: str, message: dict):
    """Handle text messages - send instructions."""
    text = message.get("text", {}).get("body", "").strip().lower()

    if text in ("hola", "hi", "hello", "ayuda", "help", "inicio", "menu"):
        await send_whatsapp_message(
            sender,
            "🌿 *Bienvenido a Spray Analyzer* 🌿\n\n"
            "Soy un bot que analiza la cobertura de rociado en hojas "
            "usando fluorescencia UV.\n\n"
            "*¿Cómo funciona?*\n"
            "1️⃣ Toma una foto de la hoja bajo luz UV\n"
            "2️⃣ Envíamela por este chat\n"
            "3️⃣ Te respondo con el análisis de cobertura\n\n"
            "📸 *¡Envía tu foto ahora!*"
        )
    else:
        await send_whatsapp_message(
            sender,
            "📸 Para analizar una hoja, envíame una *imagen*.\n\n"
            "Escribe *hola* para ver las instrucciones completas."
        )


async def handle_image_message(sender: str, message: dict):
    """Handle image messages - analyze spray coverage."""
    try:
        # Notify user we're processing
        await send_whatsapp_message(
            sender,
            "🔄 *Analizando tu imagen...*\nEsto toma unos segundos."
        )

        # Get media ID and download
        media_id = message.get("image", {}).get("id")
        if not media_id:
            await send_whatsapp_message(sender, "❌ No pude obtener la imagen. Intenta de nuevo.")
            return

        image_bytes = await download_whatsapp_media(media_id)
        logger.info(f"Downloaded image: {len(image_bytes)} bytes from {sender}")

        # Analyze with Spray API
        result = await analyze_image_via_api(image_bytes)

        coverage = result.get("coverage_percentage", 0)
        total_area = result.get("total_area", 0)
        sprayed_area = result.get("sprayed_area", 0)

        # Determine coverage quality
        if coverage >= 70:
            quality = "✅ Excelente cobertura"
        elif coverage >= 50:
            quality = "✅ Buena cobertura"
        elif coverage >= 30:
            quality = "⚠️ Cobertura moderada, considere re-aplicar"
        else:
            quality = "❌ Cobertura baja, se recomienda re-aplicar"

        # Build result text
        result_text = (
            f"🌿 *Resultado del Análisis*\n\n"
            f"📊 *Cobertura:* {coverage}%\n"
            f"📐 *Área total:* {total_area:,} px\n"
            f"💧 *Área rociada:* {sprayed_area:,} px\n\n"
            f"{quality}"
        )

        # Send processed image if available
        processed_image_b64 = result.get("processed_image")
        if processed_image_b64:
            try:
                import base64
                image_data = base64.b64decode(processed_image_b64)
                media_id = await upload_whatsapp_media(image_data, "image/png")
                await send_whatsapp_image(sender, media_id, result_text)
            except Exception as img_ex:
                logger.error(f"Error sending processed image: {img_ex}", exc_info=True)
                await send_whatsapp_message(sender, result_text)
        else:
            await send_whatsapp_message(sender, result_text)

        logger.info(f"Analysis sent to {sender}: {coverage}% coverage")

    except Exception as ex:
        logger.error(f"Error analyzing image from {sender}: {type(ex).__name__}: {ex}", exc_info=True)
        await send_whatsapp_message(
            sender,
            "❌ Hubo un error al analizar la imagen. "
            "Asegúrate de enviar una foto de hoja bajo luz UV e intenta de nuevo."
        )


# --- Webhook endpoints ---

@router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    """Webhook verification for Meta Cloud API."""
    if hub_mode == "subscribe" and hub_token == WHATSAPP_VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        return int(hub_challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook")
async def handle_webhook(request: Request):
    """Handle incoming WhatsApp messages."""
    body = await request.json()
    logger.info(f"WhatsApp webhook received")

    try:
        entry = body.get("entry", [])
        for e in entry:
            changes = e.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                messages = value.get("messages", [])

                for message in messages:
                    sender = message.get("from", "")
                    msg_type = message.get("type", "")

                    if msg_type == "image":
                        await handle_image_message(sender, message)
                    elif msg_type == "text":
                        await handle_text_message(sender, message)
                    else:
                        await send_whatsapp_message(
                            sender,
                            "🌿 *Spray Analyzer Bot*\n\n"
                            "Envíame una foto de una hoja con fluorescencia UV "
                            "y te digo el porcentaje de cobertura de spray.\n\n"
                            "📸 Solo manda la imagen!"
                        )

    except Exception as ex:
        logger.error(f"Error processing webhook: {ex}", exc_info=True)

    return JSONResponse(content={"status": "ok"}, status_code=200)
