"""
Telegram-бот с ИИ на базе Google Gemini, OpenRouter и OmniRoute.

Секреты (переменные окружения):
  - TELEGRAM_BOT_TOKEN       — токен бота от @BotFather
  - GEMINI_API_KEY           — ключ Google AI Studio (резервный)
  - OMNIRUTE_API_KEY         — ключ OmniRoute для Gemini-3.1-lite
  - OMNIRUTE_BASE_URL        — базовый URL для OmniRoute
  - OPENROUTER_API_KEY       — ключ OpenRouter для Claude-Opus, Fusion и Riverflow

Команды:
  /start    — приветствие и список команд
  /gemini   — переключиться на модель Gemini-3.1-lite (OmniRoute)
  /claude   — переключиться на модель Claude-Opus (OpenRouter)
  /think    — глубокий поиск/рассуждения (Fusion через OpenRouter)
  /image    — сгенерировать изображение (Riverflow через OpenRouter)
  /history  — просмотр и управление историей
  /settings — настройки бота
  /reset    — очистить историю диалога
  /automate — включить/выключить автоответ в текущем чате (для Telegram Business)
"""

import os
import re
import json
import time
import random
import asyncio
import logging
import tempfile
import base64
import urllib.parse
from pathlib import Path
from datetime import datetime

import httpx
from google import genai
from google.genai import types
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    BusinessConnectionHandler,
    InlineQueryHandler,
    ContextTypes,
    filters,
)

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Токены / Клиент
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

_REPLIT_BASE_URL = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")
_REPLIT_API_KEY  = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY")
_DIRECT_API_KEY  = os.environ.get("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise EnvironmentError("TELEGRAM_BOT_TOKEN не задан!")

# Официальный клиент Google GenAI (используется как резервный для текста и handle_photo)
client = None
if _REPLIT_BASE_URL and _REPLIT_API_KEY:
    client = genai.Client(
        api_key=_REPLIT_API_KEY,
        http_options=types.HttpOptions(base_url=_REPLIT_BASE_URL, api_version=""),
    )
elif _DIRECT_API_KEY:
    client = genai.Client(api_key=_DIRECT_API_KEY)
else:
    logger.warning("GEMINI_API_KEY не задан. Официальные функции Gemini будут недоступны.")

MODEL          = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MODEL_FALLBACK = os.environ.get("GEMINI_MODEL_FALLBACK", "gemini-2.0-flash")
MAX_MSG_LEN  = 4000
MAX_HISTORY  = 50   # максимум сообщений в памяти

# ---------------------------------------------------------------------------
# Вспомогательные функции официального Gemini (резервные)
# ---------------------------------------------------------------------------
QUOTA_RETRY_DELAYS = [5, 15, 30]  # секунд между попытками при квоте

def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "resource_exhausted", "quota", "rate limit", "free_cloud_budget_exceeded"))

def gemini_generate(contents, config: types.GenerateContentConfig, model: str = MODEL):
    """Вызов generate_content с повторами и фоллбэком на резервную модель при квоте."""
    if not client:
        raise ValueError("Официальный клиент Gemini не инициализирован (отсутствует GEMINI_API_KEY)")
    for attempt, delay in enumerate(QUOTA_RETRY_DELAYS + [None]):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            if _is_quota_error(exc):
                if delay is not None:
                    logger.warning("Квота модели %s превышена, жду %ds (попытка %d)...", model, delay, attempt + 1)
                    time.sleep(delay)
                    continue
                if model != MODEL_FALLBACK:
                    logger.warning("Переключаюсь на резервную модель %s", MODEL_FALLBACK)
                    return client.models.generate_content(model=MODEL_FALLBACK, contents=contents, config=config)
            raise

def gemini_stream(contents, config: types.GenerateContentConfig, model: str = MODEL):
    """Вызов generate_content_stream с повторами и фоллбэком на резервную модель при квоте."""
    if not client:
        raise ValueError("Официальный клиент Gemini не инициализирован (отсутствует GEMINI_API_KEY)")
    for attempt, delay in enumerate(QUOTA_RETRY_DELAYS + [None]):
        try:
            return client.models.generate_content_stream(model=model, contents=contents, config=config)
        except Exception as exc:
            if _is_quota_error(exc):
                if delay is not None:
                    logger.warning("Квота модели %s превышена, жду %ds (попытка %d)...", model, delay, attempt + 1)
                    time.sleep(delay)
                    continue
                if model != MODEL_FALLBACK:
                    logger.warning("Переключаюсь на резервную модель %s", MODEL_FALLBACK)
                    return client.models.generate_content_stream(model=MODEL_FALLBACK, contents=contents, config=config)
            raise

# ---------------------------------------------------------------------------
# Интеграция с OpenRouter и OmniRoute
# ---------------------------------------------------------------------------
def build_openai_payload(model: str, history: list, system_instruction: str = None, stream: bool = False, max_tokens: int = None) -> dict:
    """Форматирует историю диалога в OpenAI-совместимый Payload с поддержкой мультимодальности (изображения)."""
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    
    for h in history:
        role = "assistant" if h["role"] == "model" else "user"
        content_list = []
        
        # Разбираем части сообщения (parts)
        for p in h.get("parts", []):
            if "text" in p:
                content_list.append({
                    "type": "text",
                    "text": p["text"]
                })
            elif "inline_data" in p:
                mime_type = p["inline_data"].get("mime_type", "image/jpeg")
                data_b64 = p["inline_data"].get("data", "")
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{data_b64}"
                    }
                })
        
        # Если в контенте только текст, передаем как простую строку
        if len(content_list) == 1 and content_list[0]["type"] == "text":
            messages.append({"role": role, "content": content_list[0]["text"]})
        else:
            messages.append({"role": role, "content": content_list})
        
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload

def openrouter_omniroute_stream(url: str, headers: dict, payload: dict):
    """Синхронный генератор для стриминга ответов через SSE."""
    with httpx.Client(timeout=60.0) as client_http:
        with client_http.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                err_text = response.read().decode("utf-8", errors="ignore")
                raise Exception(f"HTTP {response.status_code}: {err_text}")
            
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        chunk = data_json["choices"][0]["delta"].get("content", "")
                        if chunk:
                            yield chunk
                    except Exception:
                        pass

async def call_external_llm(provider: str, model: str, history: list, system_instruction: str = None, stream: bool = False, max_tokens: int = None):
    """
    Вызов внешних моделей через OpenRouter или OmniRoute.
    Возвращает строку (при stream=False) или генератор чанков (при stream=True).
    """
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vanyk56/Gemini-Chatbot",
            "X-Title": "Telegram Bot",
        }
    elif provider == "omniroute":
        api_key = os.environ.get("OMNIRUTE_API_KEY")
        base_url = os.environ.get("OMNIRUTE_BASE_URL", "https://api.omniroute.online/v1")
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    else:
        raise ValueError(f"Неизвестный провайдер: {provider}")

    if not api_key:
        raise ValueError(f"Отсутствует API-ключ для провайдера {provider}!")

    payload = build_openai_payload(model, history, system_instruction, stream, max_tokens)

    if stream:
        return openrouter_omniroute_stream(url, headers, payload)
    else:
        async with httpx.AsyncClient(timeout=60.0) as client_http:
            response = await client_http.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                raise Exception(f"API {provider} returned HTTP {response.status_code}: {response.text}")
            data = response.json()
            return data["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------------
# LaTeX → читаемый текст
# ---------------------------------------------------------------------------
def latex_to_text(text: str) -> str:
    def convert_mixed(t: str) -> str:
        return re.sub(
            r"(\d+)\s+\\frac\{([^}]+)\}\{([^}]+)\}",
            lambda m: f"{m.group(1)} {m.group(2)}/{m.group(3)}",
            t,
        )

    text = re.sub(r"\$\$(.+?)\$\$", lambda m: m.group(1).strip(), text, flags=re.DOTALL)
    text = convert_mixed(text)
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", lambda m: f"{m.group(1).strip()}/{m.group(2).strip()}", text)
    text = re.sub(r"\\sqrt\{([^}]+)\}", lambda m: f"√({m.group(1).strip()})", text)

    replacements = [
        (r"\\times", "×"), (r"\\cdot", "·"), (r"\\div", "÷"),
        (r"\\pm", "±"), (r"\\leq", "≤"), (r"\\geq", "≥"),
        (r"\\neq", "≠"), (r"\\approx", "≈"), (r"\\infty", "∞"),
        (r"\\alpha", "α"), (r"\\beta", "β"), (r"\\gamma", "γ"), (r"\\pi", "π"),
        (r"\\left\(", "("), (r"\\right\)", ")"),
        (r"\\left\[", "["), (r"\\right\]", "]"),
        (r"\\,", " "), (r"\\;", " "), (r"\\!", ""),
        (r"\\quad", "  "), (r"\\qquad", "    "),
        (r"\\ldots", "..."), (r"\\cdots", "..."),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    text = re.sub(r"\^\{([^}]+)\}", lambda m: f"^{m.group(1)}", text)
    text = re.sub(r"_\{([^}]+)\}", lambda m: f"_{m.group(1)}", text)
    text = re.sub(r"\\llcorner[a-zA-Z]+", "", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\$([^$\n]+)\$", lambda m: m.group(1).strip(), text)
    return text

# ---------------------------------------------------------------------------
# Markdown → HTML для Telegram
# ---------------------------------------------------------------------------
def md_to_html(text: str) -> str:
    text = latex_to_text(text)
    text = re.sub(r"^#{1,3}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__",     r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)",       r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text

def split_message(text: str, max_len: int = MAX_MSG_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts, current = [], ""
    for paragraph in text.split("\n\n"):
        if len(current) + len(paragraph) + 2 <= max_len:
            current += ("" if not current else "\n\n") + paragraph
        else:
            if current:
                parts.append(current)
            current = paragraph
    if current:
        parts.append(current)
    return parts or [text[:max_len]]

async def send_reply(update: Update, text: str, business_connection_id: str | None = None) -> None:
    formatted = md_to_html(text)
    parts = split_message(formatted)
    for part in parts:
        kwargs: dict = {"parse_mode": ParseMode.HTML}
        if business_connection_id:
            kwargs["business_connection_id"] = business_connection_id
        try:
            await update.message.reply_text(part, **kwargs)
        except Exception:
            await update.message.reply_text(part)

# ---------------------------------------------------------------------------
# Нативный стриминг
# ---------------------------------------------------------------------------
_draft_counter = 0

def _next_draft_id() -> int:
    global _draft_counter
    _draft_counter = (_draft_counter % 2_000_000_000) + 1
    return _draft_counter

async def _send_draft(token: str, chat_id: int | str, draft_id: int, text: str) -> None:
    """Вызывает sendMessageDraft через raw Bot API."""
    url = f"https://api.telegram.org/bot{token}/sendMessageDraft"
    payload: dict = {"chat_id": chat_id, "draft_id": draft_id}
    if text:
        payload["text"] = text
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            await c.post(url, json=payload)
    except Exception as e:
        logger.debug("sendMessageDraft error: %s", e)

DRAFT_UPDATE_INTERVAL = 1.0   # секунд между обновлениями черновика
DRAFT_MIN_CHARS       = 20    # минимум символов перед первым обновлением

async def native_stream_reply(
    update: Update,
    token: str,
    stream_iter,
    *,
    reply_to_message_id: int | None = None,
) -> str:
    chat_id  = update.effective_chat.id
    draft_id = _next_draft_id()

    await _send_draft(token, chat_id, draft_id, "")

    queue: asyncio.Queue = asyncio.Queue()

    def _produce() -> None:
        try:
            for chunk in stream_iter:
                if isinstance(chunk, str):
                    txt = chunk
                else:
                    txt = getattr(chunk, "text", None) or ""
                if txt:
                    queue.put_nowait(txt)
            queue.put_nowait(None)
        except Exception as exc:
            queue.put_nowait(exc)

    asyncio.get_running_loop().run_in_executor(None, _produce)

    accumulated = ""
    last_update  = 0.0

    while True:
        try:
            chunk_text = await asyncio.wait_for(queue.get(), timeout=45.0)
        except asyncio.TimeoutError:
            break
        if chunk_text is None:
            break
        if isinstance(chunk_text, Exception):
            raise chunk_text
        accumulated += chunk_text

        now = time.monotonic()
        if now - last_update >= DRAFT_UPDATE_INTERVAL and len(accumulated) >= DRAFT_MIN_CHARS:
            await _send_draft(token, chat_id, draft_id, latex_to_text(accumulated))
            last_update = now

    if accumulated:
        final_html = md_to_html(accumulated)
        for part in split_message(final_html):
            kwargs: dict = {"parse_mode": ParseMode.HTML}
            if reply_to_message_id:
                kwargs["reply_to_message_id"] = reply_to_message_id
            try:
                await update.message.reply_text(part, **kwargs)
            except Exception:
                await update.message.reply_text(part)

    return accumulated

# ---------------------------------------------------------------------------
# Состояние пользователей (история + режим + настройки)
# ---------------------------------------------------------------------------
USER_MODE_FILE        = Path("bot/user_modes.json")
USER_SETTINGS_FILE    = Path("bot/user_settings.json")
BUSINESS_CONN_FILE    = Path("bot/business_connections.json")
PREMIUM_USERS_FILE    = Path("bot/premium_users.json")
SCHEDULED_TASKS_FILE  = Path("bot/scheduled_tasks.json")

conversation_history: dict[str, list]          = {}
conversation_timestamps: dict[str, list[str]]  = {}
user_mode: dict[int, str]                       = {}
user_settings: dict[int, dict]                  = {}
business_connections: dict[str, int]            = {}  # conn_id → user_id
user_state: dict[int, str]                      = {}  # Состояние ввода пользователя
premium_users: dict[int, dict]                  = {}  # user_id -> {"is_premium": bool, "requests": list[float]}
user_schedule_drafts: dict[int, dict]           = {}  # user_id -> черновик отложенного сообщения

def _load_scheduled_tasks() -> list:
    if SCHEDULED_TASKS_FILE.exists():
        try:
            return json.loads(SCHEDULED_TASKS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Не удалось загрузить отложенные сообщения: %s", e)
    return []

def _save_scheduled_tasks(tasks: list) -> None:
    try:
        SCHEDULED_TASKS_FILE.write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Не удалось сохранить отложенные сообщения: %s", e)


def _load_premium_users() -> None:
    global premium_users
    if PREMIUM_USERS_FILE.exists():
        try:
            data = json.loads(PREMIUM_USERS_FILE.read_text(encoding="utf-8"))
            premium_users = {int(k): v for k, v in data.items()}
            logger.info("Загружено премиум-пользователей: %d", len(premium_users))
        except Exception as e:
            logger.warning("Не удалось загрузить премиум-пользователей: %s", e)

def _save_premium_users() -> None:
    try:
        PREMIUM_USERS_FILE.write_text(
            json.dumps({str(k): v for k, v in premium_users.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Не удалось сохранить премиум-пользователей: %s", e)

def is_premium_user(user) -> bool:
    """Проверяет, является ли пользователь премиум-подписчиком или владельцем."""
    if not user:
        return False
    if user.username and user.username.lower() == "ohakol":
        return True
    user_info = premium_users.get(user.id, {})
    return user_info.get("is_premium", False)

def check_and_record_premium_request(user) -> tuple[bool, str]:
    """
    Проверяет лимиты премиум-пользователя.
    Возвращает (allow: bool, reason: str).
    """
    if not user:
        return False, "invalid_user"
    
    if user.username and user.username.lower() == "ohakol":
        return True, "owner"
    
    user_info = premium_users.get(user.id, {})
    if not user_info.get("is_premium", False):
        return False, "need_premium"
    
    # Скользящее окно: 7 дней (168 часов)
    now = time.time()
    requests = user_info.setdefault("requests", [])
    filtered_requests = [r for r in requests if now - r < 7 * 24 * 3600]
    
    if len(filtered_requests) >= 5:
        user_info["requests"] = filtered_requests
        _save_premium_users()
        return False, "limit_exceeded"
        
    filtered_requests.append(now)
    user_info["requests"] = filtered_requests
    _save_premium_users()
    return True, "allowed"

def _load_state() -> None:
    global user_mode, user_settings, business_connections
    _load_premium_users()
    if USER_MODE_FILE.exists():
        try:
            data = json.loads(USER_MODE_FILE.read_text(encoding="utf-8"))
            user_mode = {int(k): v for k, v in data.items()}
        except Exception as e:
            logger.warning("Не удалось загрузить режимы: %s", e)
    if USER_SETTINGS_FILE.exists():
        try:
            data = json.loads(USER_SETTINGS_FILE.read_text(encoding="utf-8"))
            user_settings = {int(k): v for k, v in data.items()}
        except Exception as e:
            logger.warning("Не удалось загрузить настройки: %s", e)
    if BUSINESS_CONN_FILE.exists():
        try:
            data = json.loads(BUSINESS_CONN_FILE.read_text(encoding="utf-8"))
            business_connections = {k: int(v) for k, v in data.items()}
            logger.info("Загружено бизнес-подключений: %d", len(business_connections))
        except Exception as e:
            logger.warning("Не удалось загрузить бизнес-подключения: %s", e)

def _save_business_connections() -> None:
    try:
        BUSINESS_CONN_FILE.write_text(
            json.dumps(business_connections, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Не удалось сохранить бизнес-подключения: %s", e)

def _save_modes() -> None:
    try:
        USER_MODE_FILE.write_text(
            json.dumps({str(k): v for k, v in user_mode.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Не удалось сохранить режимы: %s", e)

def _save_settings() -> None:
    try:
        USER_SETTINGS_FILE.write_text(
            json.dumps({str(k): v for k, v in user_settings.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Не удалось сохранить настройки: %s", e)

def get_settings(user_id: int) -> dict:
    return user_settings.setdefault(user_id, {
        "auto_reply": True,
        "max_history": 10,
        "language": "auto",
        "disabled_chats": [],
        "auto_reply_delay": 3.0,
    })

def set_user_mode(user_id: int, mode: str) -> None:
    user_mode[user_id] = mode
    _save_modes()

def get_history(user_id: int) -> list:
    return conversation_history.setdefault(str(user_id), [])

def get_timestamps(user_id: int) -> list:
    return conversation_timestamps.setdefault(str(user_id), [])

def add_message(user_id: int, role: str, parts_data: list) -> None:
    settings = get_settings(user_id)
    max_h = settings.get("max_history", 20)
    history = get_history(user_id)
    timestamps = get_timestamps(user_id)
    history.append({"role": role, "parts": parts_data})
    timestamps.append(datetime.now().strftime("%H:%M"))
    while len(history) > max_h * 2:
        history.pop(0)
        if timestamps:
            timestamps.pop(0)

def clear_user(user_id: int) -> None:
    conversation_history.pop(str(user_id), None)
    conversation_timestamps.pop(str(user_id), None)
    user_mode.pop(user_id, None)
    _save_modes()

# ---------------------------------------------------------------------------
# Системные промпты
# ---------------------------------------------------------------------------
_NO_LATEX = (
    "ВАЖНО: Telegram не поддерживает LaTeX. НЕ используй $, \\frac, \\times, \\sqrt и т.д. "
    "Дроби пиши как '3/4', умножение как '×', корень как '√', степень как '^2'. "
    "Используй Unicode: ×, ÷, ±, ≤, ≥, √, π."
)

SYSTEM_PROMPT_DEFAULT = (
    "Ты — дружелюбный, лаконичный и умный ИИ-собеседник SYNAPSE в Telegram. "
    "Отвечай кратко и по делу, экономно расходуй слова и избегай лишней «воды». Используй эмодзи умеренно. "
    "Пиши на том же языке, на котором пишет пользователь. "
    "Если пользователь просит тебя сгенерировать, нарисовать или создать изображение/картинку/фото, "
    "подскажи ему использовать специальную команду /image (или /generate), написав описание картинки после неё. " + _NO_LATEX
)

SYSTEM_PROMPT_IMAGE = (
    "Ты — умный ИИ-ассистент в Telegram. Пользователь прислал изображение. "
    "Внимательно проанализируй его и ответь на вопрос или опиши содержимое. "
    "Если на изображении задача или контрольная работа — реши её пошагово. "
    "Если это документ или текст — извлеки и объясни содержимое. "
    "Отвечай на языке вопроса или подписи к изображению. " + _NO_LATEX
)

def build_business_prompt(persona: str) -> str:
    base = (
        "Ты отвечаешь на сообщения от имени человека в Telegram. "
        "Пиши естественно, по-человечески — как обычный живой собеседник, не как робот. "
        "Отвечай коротко и по делу, без лишних слов. "
        "Используй разговорный стиль, иногда сленг если уместно. "
        "Не раскрывай что ты ИИ. "
        "Пиши на том же языке, что и собеседник. "
        + _NO_LATEX
    )
    if persona:
        base += (
            f"\n\nВот информация о человеке, от чьего имени ты пишешь:\n{persona}\n"
            "Учитывай эти данные при ответах на личные вопросы."
        )
    else:
        base += (
            "\n\nЕсли спрашивают что-то личное (возраст, имя, чем занимаешься) — "
            "отвечай уклончиво и естественно, как человек который не хочет об этом говорить, "
            "не как робот который 'не может ответить'."
        )
    return base

def get_main_keyboard(user_id: int) -> InlineKeyboardMarkup:
    mode = user_mode.get(user_id, "default")
    
    gemini_label = "🤖 Gemini (Бесплатно) ✅" if mode == "default" else "🤖 Gemini (Бесплатно)"
    claude_label = "👑 Claude Opus (Премиум) ✅" if mode == "claude" else "👑 Claude Opus (Премиум)"
    
    keyboard = [
        [
            InlineKeyboardButton(gemini_label, callback_data="mode:default"),
            InlineKeyboardButton(claude_label, callback_data="mode:claude"),
        ],
        [
            InlineKeyboardButton("🧠 Глубокое мышление", callback_data="action:think"),
            InlineKeyboardButton("🎨 Генерация фото", callback_data="action:image"),
        ],
        [
            InlineKeyboardButton("⚙️ Настройки", callback_data="action:settings"),
            InlineKeyboardButton("🗑️ Очистить контекст", callback_data="history:clear"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

# ---------------------------------------------------------------------------
# Логика плавающего меню
# ---------------------------------------------------------------------------
user_menu_msg: dict[int, int] = {}  # chat_id -> message_id последнего отправленного меню

async def delete_old_menu(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg_id = user_menu_msg.get(chat_id)
    if msg_id:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception as e:
            logger.debug("Failed to delete old menu message: %s", e)
        user_menu_msg.pop(chat_id, None)

async def send_new_menu(chat_id: int, context: ContextTypes.DEFAULT_TYPE, user_id: int, text: str = None) -> None:
    await delete_old_menu(chat_id, context)
    if not text:
        text = "Выберите модель или воспользуйтесь функциями ниже:"
    try:
        msg = await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=get_main_keyboard(user_id)
        )
        user_menu_msg[chat_id] = msg.message_id
    except Exception as e:
        logger.error("Failed to send new menu: %s", e)

# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    text = (
        f"Привет, {user.first_name}! 👋\n"
        "Я умный ИИ-собеседник <b>SYNAPSE</b>.\n\n"
        "Выберите модель или воспользуйтесь функциями ниже:"
    )
    await send_new_menu(update.effective_chat.id, context, user.id, text)

# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------
async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_user(update.effective_user.id)
    await update.message.reply_text("🗑️ История очищена. Начинаем с чистого листа!")

# ---------------------------------------------------------------------------
# /history
# ---------------------------------------------------------------------------
async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    history = get_history(user_id)
    timestamps = get_timestamps(user_id)

    if not history:
        await update.message.reply_text(
            "📭 История диалога пуста.\n\nНачни общение — и я запомню всё!",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🗑️ Очистить", callback_data="history:clear"),
            ]]),
        )
        return

    msg_count = len(history)
    user_msgs = sum(1 for m in history if m["role"] == "user")
    bot_msgs  = sum(1 for m in history if m["role"] == "model")

    preview_lines = []
    recent = history[-6:] if len(history) >= 6 else history
    recent_ts = timestamps[-len(recent):] if len(timestamps) >= len(recent) else timestamps
    for i, msg in enumerate(recent):
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        text_parts = msg.get("parts", [])
        text = text_parts[0].get("text", "") if text_parts else ""
        short = text[:60].replace("\n", " ")
        if len(text) > 60:
            short += "..."
        ts = recent_ts[i] if i < len(recent_ts) else ""
        preview_lines.append(f"{role_icon} <i>[{ts}]</i> {short}")

    preview = "\n".join(preview_lines)
    settings = get_settings(user_id)
    max_h = settings.get("max_history", 20)

    text = (
        f"📜 <b>История диалога</b>\n\n"
        f"💬 Всего сообщений: {msg_count} (лимит: {max_h * 2})\n"
        f"👤 Твоих: {user_msgs} | 🤖 Моих: {bot_msgs}\n\n"
        f"<b>Последние сообщения:</b>\n{preview}"
    )

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🗑️ Очистить всё", callback_data="history:clear"),
            InlineKeyboardButton("✂️ Оставить 5", callback_data="history:keep5"),
        ],
        [
            InlineKeyboardButton("📉 Лимит: 10", callback_data="settings:max_history:10"),
            InlineKeyboardButton("📊 Лимит: 20", callback_data="settings:max_history:20"),
            InlineKeyboardButton("📈 Лимит: 50", callback_data="settings:max_history:50"),
        ],
        [InlineKeyboardButton("❌ Закрыть", callback_data="menu:close")],
    ])

    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

# ---------------------------------------------------------------------------
# /settings
# ---------------------------------------------------------------------------
async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = user.id
    settings = get_settings(user_id)
    mode = user_mode.get(user_id, "default")
    auto = settings.get("auto_reply", True)
    max_h = settings.get("max_history", 20)
    persona = settings.get("persona", "")

    mode_label = {"default": "💬 Gemini (OmniRoute)", "claude": "🎭 Claude (OpenRouter)"}.get(mode, mode)
    auto_label = "✅ Вкл" if auto else "❌ Выкл"
    persona_label = (persona[:40] + "...") if len(persona) > 40 else (persona or "не задана")

    is_prem = is_premium_user(user)
    prem_status = "Активна 👑" if is_prem else "Не активна ❌"

    text = (
        "⚙️ <b>Настройки бота</b>\n\n"
        f"👑 Подписка: <b>{prem_status}</b>\n"
        f"🗂 Режим чата: <b>{mode_label}</b>\n"
        f"🔄 Авто-ответ в бизнес-чатах: <b>{auto_label}</b>\n"
        f"📝 Лимит истории: <b>{max_h} сообщений</b>\n"
        f"🎭 Личность авто-ответчика: <i>{persona_label}</i>\n\n"
        "Используйте /persona для настройки личности."
    )

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("💬 Gemini", callback_data="mode:default"),
            InlineKeyboardButton("🎭 Claude", callback_data="mode:claude"),
        ],
        [
            InlineKeyboardButton("👑 Премиум-подписка", callback_data="action:premium"),
            InlineKeyboardButton("📅 Отложенные сообщения", callback_data="settings:scheduler:menu"),
        ],
        [
            InlineKeyboardButton(
                f"🔄 Авто-ответ: {'✅' if auto else '❌'}",
                callback_data="settings:auto_reply:toggle",
            ),
        ],
        [
            InlineKeyboardButton("📝 10", callback_data="settings:max_history:10"),
            InlineKeyboardButton("📝 20", callback_data="settings:max_history:20"),
            InlineKeyboardButton("📝 50", callback_data="settings:max_history:50"),
        ],
        [
            InlineKeyboardButton("🎭 Изменить личность", callback_data="persona:prompt"),
            InlineKeyboardButton("🗑️ Сбросить личность", callback_data="persona:reset"),
        ],
        [InlineKeyboardButton("🗑️ Очистить историю", callback_data="history:clear")],
        [InlineKeyboardButton("❌ Закрыть", callback_data="menu:close")],
    ])

    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

# ---------------------------------------------------------------------------
# /persona
# ---------------------------------------------------------------------------
async def cmd_persona(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    settings = get_settings(user_id)
    persona = settings.get("persona", "")

    args = context.args
    if args:
        new_persona = " ".join(args)
        settings["persona"] = new_persona
        _save_settings()
        await update.message.reply_text(
            f"✅ <b>Личность сохранена!</b>\n\n"
            f"<i>{new_persona}</i>\n\n"
            "Теперь авто-ответчик будет использовать эти данные при личных вопросах.",
            parse_mode=ParseMode.HTML,
        )
    else:
        current = f"<i>{persona}</i>" if persona else "<i>не задана</i>"
        await update.message.reply_text(
            "🎭 <b>Настройка личности авто-ответчика</b>\n\n"
            f"Текущая: {current}\n\n"
            "Напишите после команды информацию о себе:\n\n"
            "<code>/persona Меня зовут Артём, мне 19 лет. Увлекаюсь спортом.</code>\n\n"
            "Эти данные бот будет использовать при ответах в Telegram Business.",
            parse_mode=ParseMode.HTML,
        )

# ---------------------------------------------------------------------------
# Режимы моделей
# ---------------------------------------------------------------------------
async def cmd_claude(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_premium_user(user):
        await update.message.reply_text("❌ Этот режим доступен только по Premium подписке! Оформите её с помощью /premium.")
        return
    set_user_mode(user.id, "claude")
    await update.message.reply_text("🎭 <b>Режим Claude активирован!</b>", parse_mode=ParseMode.HTML)

async def cmd_gemini(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    set_user_mode(user_id, "default")
    await update.message.reply_text("💬 <b>Режим Gemini активирован!</b>", parse_mode=ParseMode.HTML)

# ---------------------------------------------------------------------------
# Глубокие рассуждения: /think
# ---------------------------------------------------------------------------
async def handle_think_logic(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    temp_history = [{"role": "user", "parts": [{"text": query}]}]
    try:
        stream_iter = await call_external_llm(
            provider="openrouter",
            model="openrouter/fusion",
            history=temp_history,
            system_instruction="Ты — аналитический ИИ-ассистент, выполняющий детальный разбор вопросов. Отвечай подробно и по делу.",
            stream=True,
            max_tokens=3500
        )
        await native_stream_reply(update, context.bot.token, stream_iter)
    except Exception as exc:
        logger.error("Ошибка в /think: %s", exc)
        await update.message.reply_text("⚠️ Не удалось получить ответ. Попробуйте позже.")

async def cmd_think(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text(
            "✏️ Напишите вопрос после команды. Пример:\n<code>/think какая скорость света?</code>",
            parse_mode=ParseMode.HTML
        )
        return

    user = update.effective_user
    allowed, reason = check_and_record_premium_request(user)
    if not allowed:
        if reason == "need_premium":
            await update.message.reply_text("❌ Функция Глубокого мышления доступна только по Premium подписке! Оформите её с помощью /premium.")
        elif reason == "limit_exceeded":
            await update.message.reply_text("❌ Вы исчерпали лимит 5 запросов Глубокого мышления в неделю!")
        return

    await handle_think_logic(update, context, query)

# ---------------------------------------------------------------------------
# Генерация картинок: /image
# ---------------------------------------------------------------------------
async def handle_image_logic(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")

    is_russian = bool(re.search('[а-яА-Я]', prompt))
    english_prompt = prompt
    if is_russian:
        translation_prompt = (
            f"Translate the following user prompt for image generation into a detailed, high-quality English prompt. "
            f"Maintain the core style and meaning. Output ONLY the English prompt text, no explanations, no quotes, no extra text.\n\n"
            f"Prompt to translate: {prompt}"
        )
        try:
            mode = user_mode.get(user_id, "default")
            if mode == "claude":
                translation = await call_external_llm("openrouter", "anthropic/claude-opus-4.8", [{"role": "user", "parts": [{"text": translation_prompt}]}], stream=False, max_tokens=150)
            else:
                if os.environ.get("OPENROUTER_API_KEY"):
                    translation = await call_external_llm("openrouter", "google/gemini-2.5-flash-lite", [{"role": "user", "parts": [{"text": translation_prompt}]}], stream=False, max_tokens=150)
                else:
                    response = gemini_generate([{"role": "user", "parts": [{"text": translation_prompt}]}], types.GenerateContentConfig(max_output_tokens=256))
                    translation = response.text
            if translation:
                english_prompt = translation.strip()
                logger.info("Translated prompt: '%s' -> '%s'", prompt, english_prompt)
        except Exception as e:
            logger.warning("Failed to translate prompt: %s", e)

    tmp_path = None
    try:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY не настроен")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vanyk56/Gemini-Chatbot",
            "X-Title": "Telegram Bot"
        }
        payload = {
            "model": "sourceful/riverflow-v2.5-fast",
            "messages": [{"role": "user", "content": english_prompt}],
            "modalities": ["image"]
        }

        async with httpx.AsyncClient(timeout=90.0) as client_http:
            response = await client_http.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise Exception(f"OpenRouter API error {response.status_code}: {response.text}")
            
            resp_data = response.json()
            choices = resp_data.get("choices", [])
            if not choices:
                raise ValueError("No choices in OpenRouter response")
            
            msg_data = choices[0].get("message", {})
            images = msg_data.get("images", [])
            if not images:
                content = msg_data.get("content", "")
                if "data:image/" in content:
                    img_url = content
                else:
                    raise ValueError("No generated images returned in response")
            else:
                img_url = images[0].get("image_url", {}).get("url", "")

            if not img_url:
                raise ValueError("Empty image URL returned")

            # Декодируем base64 или скачиваем по http/https ссылке
            if img_url.startswith("data:image/"):
                header, base64_data = img_url.split(",", 1)
                image_bytes = base64.b64decode(base64_data)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(image_bytes)
                    tmp_path = tmp.name
            elif img_url.startswith("http"):
                async with httpx.AsyncClient(timeout=30.0) as download_client:
                    img_resp = await download_client.get(img_url)
                    if img_resp.status_code != 200:
                        raise Exception(f"Failed to download image from OpenRouter URL: {img_url}")
                    image_bytes = img_resp.content
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(image_bytes)
                    tmp_path = tmp.name
            else:
                raise ValueError(f"Unknown image URL format: {img_url}")

    except Exception as exc:
        logger.error("Ошибка генерации через OpenRouter: %s", exc)
        # Резервный Pollinations AI
        logger.info("Переключение на Pollinations AI...")
        try:
            encoded_prompt = urllib.parse.quote(english_prompt)
            pollinations_url = f"https://image.pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&nologo=true"
            async with httpx.AsyncClient(timeout=60.0) as client_http:
                response = await client_http.get(pollinations_url)
                if response.status_code != 200:
                    raise Exception(f"Pollinations AI returned {response.status_code}")
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name
        except Exception as exc2:
            logger.error("Ошибка Pollinations AI: %s", exc2)
            await update.message.reply_text("⚠️ Не удалось сгенерировать изображение.")
            return

    try:
        with open(tmp_path, "rb") as f:
            await update.message.reply_photo(
                photo=f,
                caption=f"🎨 <b>Запрос:</b> {prompt}\n🇬🇧 <b>Промпт:</b> {english_prompt}",
                parse_mode=ParseMode.HTML
            )
    except Exception as e:
        logger.error("Ошибка отправки фото: %s", e)
        await update.message.reply_text("⚠️ Ошибка при отправке изображения.")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = " ".join(context.args).strip()
    if not prompt:
        await update.message.reply_text("✏️ Напишите описание картинки.", parse_mode=ParseMode.HTML)
        return
    await handle_image_logic(update, context, prompt)

# ---------------------------------------------------------------------------
# Интеграция с Telegram Business: включение/выключение чатов
# ---------------------------------------------------------------------------
async def cmd_automate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    settings = get_settings(user_id)
    disabled = settings.setdefault("disabled_chats", [])
    
    # Если пишем в ЛС с ботом, показываем список отключенных
    if update.effective_chat.type == "private":
        if not disabled:
            await update.message.reply_text(
                "ℹ️ Автоответ включен для всех ваших чатов.\n"
                "Напишите <code>/automate</code> прямо в любом бизнес-чате, чтобы отключить его для этого чата.",
                parse_mode=ParseMode.HTML
            )
        else:
            await update.message.reply_text(
                f"ℹ️ У вас отключен автоответ для {len(disabled)} чатов.\n"
                "Напишите <code>/automate</code> в том чате, где хотите снова его включить.",
                parse_mode=ParseMode.HTML
            )
    else:
        # В группе/супергруппе (если бот запущен)
        chat_id = update.effective_chat.id
        if chat_id in disabled:
            disabled.remove(chat_id)
            _save_settings()
            await update.message.reply_text("✅ <b>Автоответ для этого чата включен!</b>", parse_mode=ParseMode.HTML)
        else:
            disabled.append(chat_id)
            _save_settings()
            await update.message.reply_text("❌ <b>Автоответ для этого чата выключен!</b>", parse_mode=ParseMode.HTML)

# ---------------------------------------------------------------------------
# /premium
# ---------------------------------------------------------------------------
async def cmd_premium(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user = update.effective_user
    
    if update.effective_chat.type == "private":
        await delete_old_menu(chat_id, context)

    # Проверяем текущий статус подписки
    is_prem = is_premium_user(user)
    status_text = "👑 <b>Активна</b>" if is_prem else "❌ <b>Не активна</b>"
    
    # Считаем количество оставшихся запросов на этой неделе
    if user.username and user.username.lower() == "ohakol":
        limit_text = "Бесконечно запросов (Владелец)"
    elif is_prem:
        user_info = premium_users.get(user_id, {})
        now = time.time()
        requests = user_info.get("requests", [])
        filtered_requests = [r for r in requests if now - r < 7 * 24 * 3600]
        remaining = max(0, 5 - len(filtered_requests))
        limit_text = f"{remaining} из 5 запросов осталось на этой неделе"
    else:
        limit_text = "0 из 5 запросов доступно (требуется Premium)"

    text = (
        "👑 <b>SYNAPSE Premium</b>\n\n"
        f"Статус вашей подписки: {status_text}\n"
        f"Ваш лимит: <b>{limit_text}</b>\n\n"
        "Премиум-подписка открывает доступ к:\n"
        "1. 🎭 Режиму <b>Claude Opus (Премиум)</b>\n"
        "2. 🧠 Функции <b>Глубокого мышления</b>\n\n"
        "Лимит для обычных премиум-пользователей составляет <b>5 запросов в неделю</b>."
    )

    keyboard_buttons = []
    if not is_prem:
        keyboard_buttons.append([InlineKeyboardButton("👑 Активировать демо-подписку", callback_data="premium:activate")])
    keyboard_buttons.append([InlineKeyboardButton("❌ Закрыть", callback_data="menu:close")])
    keyboard = InlineKeyboardMarkup(keyboard_buttons)

    if update.effective_chat.type == "private":
        msg = await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        user_menu_msg[chat_id] = msg.message_id
    else:
        await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

# ---------------------------------------------------------------------------
# Отложенные сообщения по расписанию: /schedule
# ---------------------------------------------------------------------------
async def cmd_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_premium_user(user):
        await update.message.reply_text("❌ Планирование сообщений доступно только по Premium подписке!")
        return

    args = context.args
    if len(args) < 3:
        await update.message.reply_text(
            "✏️ Использование: <code>/schedule [ДД.ММ.ГГГГ] [ЧЧ:ММ] [Текст сообщения]</code>\n"
            "Пример: <code>/schedule 24.06.2026 10:00 Привет, это запланированное сообщение!</code>",
            parse_mode=ParseMode.HTML
        )
        return

    date_str = args[0]
    time_str = args[1]
    msg_text = " ".join(args[2:])

    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M")
    except ValueError:
        await update.message.reply_text("❌ Неверный формат даты/времени. Используйте формат: ДД.ММ.ГГГГ ЧЧ:ММ.")
        return

    now = datetime.now()
    if dt <= now:
        await update.message.reply_text("❌ Время отправки должно быть в будущем.")
        return

    chat_id = update.effective_chat.id
    conn_id = None

    if update.business_message:
        conn_id = update.business_message.business_connection_id
    elif update.message and update.message.business_connection_id:
        conn_id = update.message.business_connection_id

    if not conn_id:
        for cid, uid in business_connections.items():
            if uid == user.id:
                conn_id = cid
                break

    task_id = f"task_{int(dt.timestamp())}_{random.randint(1000, 9999)}"
    task_data = {
        "id": task_id,
        "chat_id": chat_id,
        "conn_id": conn_id,
        "text": msg_text,
        "run_at": dt.isoformat(),
        "creator_id": user.id
    }

    tasks = _load_scheduled_tasks()
    tasks.append(task_data)
    _save_scheduled_tasks(tasks)

    if context.job_queue:
        context.job_queue.run_once(
            send_scheduled_message_callback,
            when=dt,
            data=task_id,
            name=task_id,
            chat_id=chat_id
        )

    await update.message.reply_text(f"📅 Сообщение успешно запланировано на {date_str} {time_str}!")

# ---------------------------------------------------------------------------
# Административные команды владельца (@ohakol)
# ---------------------------------------------------------------------------
async def cmd_grant_premium(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user.username or user.username.lower() != "ohakol":
        await update.message.reply_text("❌ У вас нет прав для выполнения этой команды.")
        return

    target_user_id = None
    # 1. Попытка получить ID из reply
    if update.message.reply_to_message:
        target_user_id = update.message.reply_to_message.from_user.id
    # 2. Попытка получить ID из аргументов
    elif context.args:
        try:
            target_user_id = int(context.args[0])
        except ValueError:
            pass

    if not target_user_id:
        await update.message.reply_text(
            "✏️ Использование: отправьте команду в ответ на сообщение пользователя или укажите его ID:\n"
            "<code>/grant_premium [user_id]</code>",
            parse_mode=ParseMode.HTML
        )
        return

    user_info = premium_users.setdefault(target_user_id, {})
    user_info["is_premium"] = True
    _save_premium_users()

    await update.message.reply_text(f"👑 Пользователю <code>{target_user_id}</code> успешно выдан статус <b>Premium</b>!", parse_mode=ParseMode.HTML)

async def cmd_revoke_premium(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user.username or user.username.lower() != "ohakol":
        await update.message.reply_text("❌ У вас нет прав для выполнения этой команды.")
        return

    target_user_id = None
    if update.message.reply_to_message:
        target_user_id = update.message.reply_to_message.from_user.id
    elif context.args:
        try:
            target_user_id = int(context.args[0])
        except ValueError:
            pass

    if not target_user_id:
        await update.message.reply_text(
            "✏️ Использование: отправьте команду в ответ на сообщение пользователя или укажите его ID:\n"
            "<code>/revoke_premium [user_id]</code>",
            parse_mode=ParseMode.HTML
        )
        return

    if target_user_id in premium_users:
        premium_users[target_user_id]["is_premium"] = False
        _save_premium_users()
        await update.message.reply_text(f"🚫 У пользователя <code>{target_user_id}</code> успешно аннулирован статус <b>Premium</b>.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(f"❓ Пользователь <code>{target_user_id}</code> не найден в списке премиум-пользователей.", parse_mode=ParseMode.HTML)

async def cmd_list_premium(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user.username or user.username.lower() != "ohakol":
        await update.message.reply_text("❌ У вас нет прав для выполнения этой команды.")
        return

    active_premiums = [k for k, v in premium_users.items() if v.get("is_premium")]
    if not active_premiums:
        await update.message.reply_text("📭 Список премиум-пользователей пуст.")
        return

    lines = []
    for uid in active_premiums:
        req_count = len(premium_users[uid].get("requests", []))
        lines.append(f"• ID: <code>{uid}</code> (Запросов за неделю: {req_count})")

    text = "👑 <b>Список премиум-пользователей:</b>\n\n" + "\n".join(lines)
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

# ---------------------------------------------------------------------------
# Callback-кнопки (история / настройки / режим)
# ---------------------------------------------------------------------------
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    if data == "settings:scheduler:menu":
        user_state.pop(user_id, None)
        user_schedule_drafts.pop(user_id, None)
        
        tasks = _load_scheduled_tasks()
        user_tasks = [t for t in tasks if t.get("creator_id") == user_id]
        
        text = (
            "📅 <b>Управление отложенными сообщениями</b>\n\n"
            f"Запланировано сообщений: <b>{len(user_tasks)}</b>\n\n"
            "Здесь вы можете запланировать автоматическую отправку сообщения в любой чат "
            "(включая бизнес-чаты) в выбранное время."
        )
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🆕 Запланировать новое", callback_data="settings:scheduler:create")],
            [InlineKeyboardButton("📋 Список активных", callback_data="settings:scheduler:list")],
            [InlineKeyboardButton("⬅️ Назад в настройки", callback_data="action:settings")]
        ])
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        await query.answer()
        return

    if data == "settings:scheduler:create":
        if not is_premium_user(query.from_user):
            await query.answer("❌ Планирование сообщений доступно только по Premium подписке!", show_alert=True)
            return
            
        user_state[user_id] = "awaiting_schedule_chat"
        user_schedule_drafts[user_id] = {}
        
        text = (
            "📅 <b>Новое отложенное сообщение (Шаг 1 из 3)</b>\n\n"
            "<b>Укажите получателя.</b>\n"
            "Вы можете:\n"
            "• Написать ID чата (например, <code>-10023456789</code> или ID пользователя)\n"
            "• Написать юзернейм (например, <code>@username</code>)\n"
            "• Переслать любое сообщение от пользователя/группы в этот чат с ботом\n\n"
            "Отправьте данные следующим сообщением."
        )
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("❌ Отмена", callback_data="settings:scheduler:menu")
        ]])
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        await query.answer()
        return

    if data == "settings:scheduler:list":
        tasks = _load_scheduled_tasks()
        user_tasks = [t for t in tasks if t.get("creator_id") == user_id]
        
        if not user_tasks:
            text = "📋 <b>У вас нет активных отложенных сообщений.</b>"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🆕 Запланировать", callback_data="settings:scheduler:create")],
                [InlineKeyboardButton("⬅️ Назад", callback_data="settings:scheduler:menu")]
            ])
            await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
            await query.answer()
            return
            
        lines = []
        keyboard_buttons = []
        for i, t in enumerate(user_tasks[:5]):
            dt = datetime.fromisoformat(t["run_at"])
            dt_str = dt.strftime("%d.%m.%Y %H:%M")
            preview = t["text"][:30] + ("..." if len(t["text"]) > 30 else "")
            lines.append(
                f"<b>{i+1}.</b> Получатель: <code>{t['chat_id']}</code>\n"
                f"Время: <b>{dt_str}</b>\n"
                f"Текст: <i>{preview}</i>\n"
            )
            keyboard_buttons.append([
                InlineKeyboardButton(f"❌ Удалить #{i+1}", callback_data=f"scheduler:delete:{t['id']}")
            ])
            
        text = "📋 <b>Список отложенных сообщений:</b>\n\n" + "\n".join(lines)
        if len(user_tasks) > 5:
            text += f"\n<i>Показано 5 из {len(user_tasks)} задач.</i>"
            
        keyboard_buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="settings:scheduler:menu")])
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard_buttons))
        await query.answer()
        return

    if data.startswith("scheduler:delete:"):
        task_id = data.split(":")[2]
        
        tasks = _load_scheduled_tasks()
        tasks = [t for t in tasks if t["id"] != task_id]
        _save_scheduled_tasks(tasks)
        
        if context.job_queue:
            jobs = context.job_queue.get_jobs_by_name(task_id)
            for j in jobs:
                j.schedule_removal()
                
        await query.answer("Сообщение удалено!")
        tasks = _load_scheduled_tasks()
        user_tasks = [t for t in tasks if t.get("creator_id") == user_id]
        
        if not user_tasks:
            text = "📋 <b>У вас нет активных отложенных сообщений.</b>"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🆕 Запланировать", callback_data="settings:scheduler:create")],
                [InlineKeyboardButton("⬅️ Назад", callback_data="settings:scheduler:menu")]
            ])
            await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
            return
            
        lines = []
        keyboard_buttons = []
        for i, t in enumerate(user_tasks[:5]):
            dt = datetime.fromisoformat(t["run_at"])
            dt_str = dt.strftime("%d.%m.%Y %H:%M")
            preview = t["text"][:30] + ("..." if len(t["text"]) > 30 else "")
            lines.append(
                f"<b>{i+1}.</b> Получатель: <code>{t['chat_id']}</code>\n"
                f"Время: <b>{dt_str}</b>\n"
                f"Текст: <i>{preview}</i>\n"
            )
            keyboard_buttons.append([
                InlineKeyboardButton(f"❌ Удалить #{i+1}", callback_data=f"scheduler:delete:{t['id']}")
            ])
            
        text = "📋 <b>Список отложенных сообщений:</b>\n\n" + "\n".join(lines)
        if len(user_tasks) > 5:
            text += f"\n<i>Показано 5 из {len(user_tasks)} задач.</i>"
            
        keyboard_buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="settings:scheduler:menu")])
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard_buttons))
        return

    if data == "menu:close":
        await query.message.delete()
        await query.answer()
        return

    if data.startswith("premium:activate"):
        user_info = premium_users.setdefault(user_id, {})
        user_info["is_premium"] = True
        user_info["requests"] = []
        _save_premium_users()
        
        await query.answer("Премиум-подписка активирована!")
        
        status_text = "👑 <b>Активна</b>"
        limit_text = "5 из 5 запросов осталось на этой неделе"
        
        text = (
            "👑 <b>SYNAPSE Premium</b>\n\n"
            f"Статус вашей подписки: {status_text}\n"
            f"Ваш лимит: <b>{limit_text}</b>\n\n"
            "Премиум-подписка открывает доступ к:\n"
            "1. 🎭 Режиму <b>Claude Opus (Премиум)</b>\n"
            "2. 🧠 Функции <b>Глубокого мышления</b>\n\n"
            "Лимит для обычных премиум-пользователей составляет <b>5 запросов в неделю</b>."
        )
        
        if "settings" in data:
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("⬅️ Назад в настройки", callback_data="action:settings")]
            ])
        else:
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Закрыть", callback_data="menu:close")]
            ])
            
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        return

    if data == "action:premium":
        user = query.from_user
        is_prem = is_premium_user(user)
        status_text = "👑 <b>Активна</b>" if is_prem else "❌ <b>Не активна</b>"
        if user.username and user.username.lower() == "ohakol":
            limit_text = "Бесконечно запросов (Владелец)"
        elif is_prem:
            user_info = premium_users.get(user_id, {})
            now = time.time()
            requests = user_info.get("requests", [])
            filtered_requests = [r for r in requests if now - r < 7 * 24 * 3600]
            remaining = max(0, 5 - len(filtered_requests))
            limit_text = f"{remaining} из 5 запросов осталось на этой неделе"
        else:
            limit_text = "0 из 5 запросов доступно (требуется Premium)"

        text = (
            "👑 <b>SYNAPSE Premium</b>\n\n"
            f"Статус вашей подписки: {status_text}\n"
            f"Ваш лимит: <b>{limit_text}</b>\n\n"
            "Премиум-подписка открывает доступ к:\n"
            "1. 🎭 Режиму <b>Claude Opus (Премиум)</b>\n"
            "2. 🧠 Функции <b>Глубокого мышления</b>\n\n"
            "Лимит для обычных премиум-пользователей составляет <b>5 запросов в неделю</b>."
        )

        keyboard_buttons = []
        if not is_prem:
            keyboard_buttons.append([InlineKeyboardButton("👑 Активировать демо-подписку", callback_data="premium:activate:settings")])
        keyboard_buttons.append([InlineKeyboardButton("⬅️ Назад в настройки", callback_data="action:settings")])
        keyboard = InlineKeyboardMarkup(keyboard_buttons)
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        await query.answer()
        return

    if data == "menu:back":
        user_state.pop(user_id, None)
        user = query.from_user
        text = (
            f"Привет, {user.first_name}! 👋\n"
            "Я умный ИИ-собеседник <b>SYNAPSE</b>.\n\n"
            "Выберите модель или воспользуйтесь функциями ниже:"
        )
        await query.edit_message_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=get_main_keyboard(user_id)
        )
        await query.answer()
        return

    if data == "action:think":
        user = query.from_user
        if not is_premium_user(user):
            await query.answer("❌ Глубокое мышление доступно только по Premium подписке!", show_alert=True)
            return
            
        user_state[user_id] = "awaiting_think"
        await query.edit_message_text(
            "🧠 <b>Режим глубокого мышления активирован.</b>\n\n"
            "Отправьте ваш вопрос следующим сообщением (без команды /think).",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("⬅️ Назад в меню", callback_data="menu:back")
            ]])
        )
        await query.answer()
        return

    if data == "action:image":
        user_state[user_id] = "awaiting_image"
        await query.edit_message_text(
            "🎨 <b>Режим генерации изображений активирован.</b>\n\n"
            "Отправьте описание картинки следующим сообщением (без команды /image).",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("⬅️ Назад в меню", callback_data="menu:back")
            ]])
        )
        await query.answer()
        return

    if data == "action:settings":
        user = query.from_user
        is_prem = is_premium_user(user)
        prem_status = "Активна 👑" if is_prem else "Не активна ❌"

        settings = get_settings(user_id)
        mode = user_mode.get(user_id, "default")
        auto = settings.get("auto_reply", True)
        max_h = settings.get("max_history", 10)
        persona = settings.get("persona", "")

        mode_label = {"default": "Gemini (Бесплатно)", "claude": "Claude Opus (Премиум)"}.get(mode, mode)
        auto_label = "✅ Вкл" if auto else "❌ Выкл"
        persona_label = (persona[:40] + "...") if len(persona) > 40 else (persona or "не задана")

        text = (
            "⚙️ <b>Настройки бота SYNAPSE</b>\n\n"
            f"👑 Подписка: <b>{prem_status}</b>\n"
            f"🗂 Режим чата: <b>{mode_label}</b>\n"
            f"🔄 Авто-ответ в бизнес-чатах: <b>{auto_label}</b>\n"
            f"📝 Лимит истории: <b>{max_h} сообщений</b>\n"
            f"🎭 Личность авто-ответчика: <i>{persona_label}</i>\n\n"
            "Используйте /persona для настройки личности."
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("👑 Премиум-подписка", callback_data="action:premium"),
                InlineKeyboardButton("📅 Отложенные сообщения", callback_data="settings:scheduler:menu"),
            ],
            [
                InlineKeyboardButton(f"🔄 Авто-ответ: {'✅' if auto else '❌'}", callback_data="settings:auto_reply:toggle"),
            ],
            [
                InlineKeyboardButton("📝 История: 10", callback_data="settings:max_history:10"),
                InlineKeyboardButton("📝 История: 20", callback_data="settings:max_history:20"),
                InlineKeyboardButton("📝 История: 50", callback_data="settings:max_history:50"),
            ],
            [
                InlineKeyboardButton("🎭 Изменить личность", callback_data="persona:prompt"),
                InlineKeyboardButton("🗑️ Сбросить личность", callback_data="persona:reset"),
            ],
            [InlineKeyboardButton("⬅️ Назад в меню", callback_data="menu:back")],
        ])

        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        await query.answer()
        return

    if data == "history:clear":
        clear_user(user_id)
        await query.answer("История очищена!")
        user = query.from_user
        text = (
            f"Привет, {user.first_name}! 👋\n"
            "Я умный ИИ-собеседник <b>SYNAPSE</b>.\n\n"
            "Выберите модель или воспользуйтесь функциями ниже:"
        )
        await query.edit_message_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=get_main_keyboard(user_id)
        )
        return

    if data == "history:keep5":
        history = get_history(user_id)
        timestamps = get_timestamps(user_id)
        if len(history) > 10:
            conversation_history[str(user_id)] = history[-10:]
            conversation_timestamps[str(user_id)] = timestamps[-10:]
        await query.answer("Оставлены последние 5 обменов.")
        return

    if data.startswith("mode:"):
        new_mode = data.split(":")[1]
        if new_mode == "claude" and not is_premium_user(query.from_user):
            await query.answer("❌ Claude Opus доступен только по Premium подписке!", show_alert=True)
            return
            
        conversation_history.pop(str(user_id), None)
        set_user_mode(user_id, new_mode)
        label = {"default": "Gemini (Бесплатно)", "claude": "Claude Opus (Премиум)"}.get(new_mode, new_mode)
        
        await query.edit_message_reply_markup(reply_markup=get_main_keyboard(user_id))
        await query.answer(f"Режим изменен на {label}")
        return

    if data.startswith("settings:"):
        parts = data.split(":")
        key = parts[1]
        val = parts[2] if len(parts) > 2 else None
        settings = get_settings(user_id)

        if key == "auto_reply" and val == "toggle":
            settings["auto_reply"] = not settings.get("auto_reply", True)
            _save_settings()
            status = "включен" if settings["auto_reply"] else "выключен"
            await query.answer(f"Авто-ответ {status}.")
        elif key == "max_history" and val:
            settings["max_history"] = int(val)
            _save_settings()
            await query.answer(f"Лимит истории: {val} сообщений.")

        user = query.from_user
        is_prem = is_premium_user(user)
        prem_status = "Активна 👑" if is_prem else "Не активна ❌"

        mode = user_mode.get(user_id, "default")
        auto = settings.get("auto_reply", True)
        max_h = settings.get("max_history", 10)
        persona = settings.get("persona", "")
        mode_label = {"default": "Gemini (Бесплатно)", "claude": "Claude Opus (Премиум)"}.get(mode, mode)
        auto_label = "✅ Вкл" if auto else "❌ Выкл"
        persona_label = (persona[:40] + "...") if len(persona) > 40 else (persona or "не задана")

        text = (
            "⚙️ <b>Настройки бота SYNAPSE</b>\n\n"
            f"👑 Подписка: <b>{prem_status}</b>\n"
            f"🗂 Режим чата: <b>{mode_label}</b>\n"
            f"🔄 Авто-ответ в бизнес-чатах: <b>{auto_label}</b>\n"
            f"📝 Лимит истории: <b>{max_h} сообщений</b>\n"
            f"🎭 Личность авто-ответчика: <i>{persona_label}</i>\n\n"
            "Используйте /persona для настройки личности."
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("👑 Премиум-подписка", callback_data="action:premium"),
                InlineKeyboardButton("📅 Отложенные сообщения", callback_data="settings:scheduler:menu"),
            ],
            [
                InlineKeyboardButton(f"🔄 Авто-ответ: {'✅' if auto else '❌'}", callback_data="settings:auto_reply:toggle"),
            ],
            [
                InlineKeyboardButton("📝 История: 10", callback_data="settings:max_history:10"),
                InlineKeyboardButton("📝 История: 20", callback_data="settings:max_history:20"),
                InlineKeyboardButton("📝 История: 50", callback_data="settings:max_history:50"),
            ],
            [
                InlineKeyboardButton("🎭 Изменить личность", callback_data="persona:prompt"),
                InlineKeyboardButton("🗑️ Сбросить личность", callback_data="persona:reset"),
            ],
            [InlineKeyboardButton("⬅️ Назад в меню", callback_data="menu:back")],
        ])
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        return

    if data.startswith("persona:"):
        action = data.split(":")[1]
        settings = get_settings(user_id)
        if action == "reset":
            settings.pop("persona", None)
            _save_settings()
            await query.answer("Личность сброшена.")
        elif action == "prompt":
            await query.edit_message_text(
                "🎭 Отправьте команду с описанием себя:\n\n"
                "<code>/persona Меня зовут Артём, мне 19 лет. Учусь в универе, увлекаюсь музыкой.</code>",
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("⬅️ Назад в настройки", callback_data="action:settings")
                ]])
            )
            await query.answer()
            return

        user = query.from_user
        is_prem = is_premium_user(user)
        prem_status = "Активна 👑" if is_prem else "Не активна ❌"

        mode = user_mode.get(user_id, "default")
        auto = settings.get("auto_reply", True)
        max_h = settings.get("max_history", 10)
        persona = settings.get("persona", "")
        mode_label = {"default": "Gemini (Бесплатно)", "claude": "Claude Opus (Премиум)"}.get(mode, mode)
        auto_label = "✅ Вкл" if auto else "❌ Выкл"
        persona_label = (persona[:40] + "...") if len(persona) > 40 else (persona or "не задана")

        text = (
            "⚙️ <b>Настройки бота SYNAPSE</b>\n\n"
            f"👑 Подписка: <b>{prem_status}</b>\n"
            f"🗂 Режим чата: <b>{mode_label}</b>\n"
            f"🔄 Авто-ответ в бизнес-чатах: <b>{auto_label}</b>\n"
            f"📝 Лимит истории: <b>{max_h} сообщений</b>\n"
            f"🎭 Личность авто-ответчика: <i>{persona_label}</i>\n\n"
            "Используйте /persona для настройки личности."
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("👑 Премиум-подписка", callback_data="action:premium"),
                InlineKeyboardButton("📅 Отложенные сообщения", callback_data="settings:scheduler:menu"),
            ],
            [
                InlineKeyboardButton(f"🔄 Авто-ответ: {'✅' if auto else '❌'}", callback_data="settings:auto_reply:toggle"),
            ],
            [
                InlineKeyboardButton("📝 История: 10", callback_data="settings:max_history:10"),
                InlineKeyboardButton("📝 История: 20", callback_data="settings:max_history:20"),
                InlineKeyboardButton("📝 История: 50", callback_data="settings:max_history:50"),
            ],
            [
                InlineKeyboardButton("🎭 Изменить личность", callback_data="persona:prompt"),
                InlineKeyboardButton("🗑️ Сбросить личность", callback_data="persona:reset"),
            ],
            [InlineKeyboardButton("⬅️ Назад в меню", callback_data="menu:back")],
        ])
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        return

# ---------------------------------------------------------------------------
# Business Bot: подключение / отключение
# ---------------------------------------------------------------------------
async def handle_business_connection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conn = update.business_connection
    user_id = conn.user.id

    if conn.is_enabled:
        business_connections[conn.id] = user_id
        _save_business_connections()
        logger.info("Business подключение: user=%d conn_id=%s", user_id, conn.id)
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=(
                    "✅ <b>Автоматизация чатов подключена!</b>\n\n"
                    "Теперь я буду отвечать на сообщения в твоих чатах от твоего имени.\n\n"
                    "⚙️ Управляй настройками через /settings\n"
                    "🔄 Авто-ответ можно отключить там же.\n"
                    "🚫 Для отключения автоответа в конкретном чате, напиши <code>/automate</code> прямо в нём."
                ),
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.warning("Не удалось отправить уведомление: %s", e)
    else:
        business_connections.pop(conn.id, None)
        _save_business_connections()
        logger.info("Business отключено: user=%d conn_id=%s", user_id, conn.id)
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text="❌ Автоматизация чатов отключена.",
            )
        except Exception as e:
            logger.warning("Не удалось отправить уведомление: %s", e)

# ---------------------------------------------------------------------------
# Business Bot: входящие сообщения через бизнес-аккаунт
# ---------------------------------------------------------------------------
async def handle_business_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.business_message
    if not msg or not msg.text:
        return

    conn_id = msg.business_connection_id
    owner_id = business_connections.get(conn_id)

    if owner_id is None:
        try:
            conn = await context.bot.get_business_connection(conn_id)
            owner_id = conn.user.id
            business_connections[conn_id] = owner_id
            _save_business_connections()
            logger.info("Восстановлено business подключение: user=%d conn_id=%s", owner_id, conn_id)
        except Exception as e:
            logger.warning("Не удалось получить business подключение %s: %s", conn_id, e)
            return

    owner_settings = get_settings(owner_id)
    if not owner_settings.get("auto_reply", True):
        return

    # Проверяем, не отключен ли автоответ для этого чата
    disabled_chats = owner_settings.setdefault("disabled_chats", [])
    
    # Если сам владелец пишет команду /automate в бизнес-чате
    if msg.from_user and msg.from_user.id == owner_id:
        if msg.text.strip().startswith("/automate"):
            if msg.chat.id in disabled_chats:
                disabled_chats.remove(msg.chat.id)
                _save_settings()
                await context.bot.send_message(
                    chat_id=msg.chat.id,
                    text="✅ <b>Автоответ для этого чата включен!</b>",
                    parse_mode=ParseMode.HTML,
                    business_connection_id=conn_id
                )
            else:
                disabled_chats.append(msg.chat.id)
                _save_settings()
                await context.bot.send_message(
                    chat_id=msg.chat.id,
                    text="❌ <b>Автоответ для этого чата выключен!</b>",
                    parse_mode=ParseMode.HTML,
                    business_connection_id=conn_id
                )
        return

    # Если автоответ в этом чате выключен владельцем
    if msg.chat.id in disabled_chats:
        return

    sender_name = msg.from_user.first_name if msg.from_user else "Собеседник"
    logger.info("Business сообщение от %s (conn=%s): %s", sender_name, conn_id, msg.text[:50])

    # Имитируем человеческое поведение: задержка + статус печатает...
    delay = float(owner_settings.get("auto_reply_delay", 3.0))
    actual_delay = random.uniform(0.7 * delay, 1.3 * delay)

    async def send_typing_loop():
        try:
            steps = int(actual_delay / 4.0) + 1
            for _ in range(steps):
                await context.bot.send_chat_action(
                    chat_id=msg.chat.id,
                    action="typing",
                    business_connection_id=conn_id,
                )
                await asyncio.sleep(min(4.0, actual_delay))
        except Exception:
            pass

    await asyncio.gather(send_typing_loop(), asyncio.sleep(actual_delay))

    key = f"biz_{conn_id}_{msg.chat.id}"
    biz_history = conversation_history.setdefault(key, [])
    biz_history.append({"role": "user", "parts": [{"text": msg.text}]})

    try:
        persona = owner_settings.get("persona", "")
        system_prompt = build_business_prompt(persona)

        # Для автоматизации используем google/gemini-2.5-flash-lite через OpenRouter
        if os.environ.get("OPENROUTER_API_KEY"):
            reply_text = await call_external_llm(
                provider="openrouter",
                model="google/gemini-2.5-flash-lite",
                history=biz_history,
                system_instruction=system_prompt,
                stream=False,
                max_tokens=1024
            )
        else:
            response = gemini_generate(
                contents=biz_history,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1024,
                ),
            )
            reply_text = response.text
    except Exception as exc:
        logger.error("Ошибка при генерации автоответа в бизнес-чате: %s", exc)
        return

    biz_history.append({"role": "model", "parts": [{"text": reply_text}]})
    if len(biz_history) > 10:
        conversation_history[key] = biz_history[-10:]

    formatted = md_to_html(reply_text)
    for part in split_message(formatted):
        try:
            await context.bot.send_message(
                chat_id=msg.chat.id,
                text=part,
                parse_mode=ParseMode.HTML,
                business_connection_id=conn_id,
            )
        except Exception as e:
            logger.error("Ошибка отправки business-ответа: %s", e)

# ---------------------------------------------------------------------------
# Обработка изображений ( handle_photo ) с полной поддержкой OmniRoute & OpenRouter
# ---------------------------------------------------------------------------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = user.id
    mode = user_mode.get(user_id, "default")
    if mode == "claude":
        allowed, reason = check_and_record_premium_request(user)
        if not allowed:
            if reason == "need_premium":
                await update.message.reply_text("❌ Режим Claude доступен только по Premium подписке! Переключаю вас на Gemini.")
            elif reason == "limit_exceeded":
                await update.message.reply_text("❌ Вы исчерпали лимит 5 премиум-запросов в неделю! Переключаю вас на Gemini.")
            set_user_mode(user_id, "default")
            mode = "default"

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    caption = update.message.caption or ""
    user_prompt = caption if caption else "Проанализируй это изображение и опиши что на нём."

    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
        mime_type = "image/jpeg"
    elif update.message.document and update.message.document.mime_type.startswith("image/"):
        photo_file = await update.message.document.get_file()
        mime_type = update.message.document.mime_type
    else:
        await update.message.reply_text("⚠️ Не удалось получить изображение.")
        return

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    await photo_file.download_to_drive(tmp_path)

    try:
        # Сжимаем и уменьшаем разрешение изображения с помощью Pillow для экономии токенов
        from PIL import Image
        from io import BytesIO
        try:
            with Image.open(tmp_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                # Уменьшаем разрешение, если длинная сторона больше 1024px (сохраняя пропорции)
                width, height = img.size
                if max(width, height) > 1024:
                    if width > height:
                        new_width = 1024
                        new_height = int(height * (1024 / width))
                    else:
                        new_height = 1024
                        new_width = int(width * (1024 / height))
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                out = BytesIO()
                img.save(out, format="JPEG", quality=80)
                image_bytes = out.getvalue()
                mime_type = "image/jpeg"
        except Exception as compress_err:
            logger.warning("Ошибка при сжатии изображения: %s. Используем оригинал.", compress_err)
            with open(tmp_path, "rb") as f:
                image_bytes = f.read()

        base64_data = base64.b64encode(image_bytes).decode()
        
        # Подготавливаем блок inline_data (картинка) + текстовый запрос
        photo_part = {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64_data
            }
        }
        text_part = {"text": user_prompt}
        
        # Объединяем с историей сообщений пользователя
        history = get_history(user_id)
        current_request = history + [{"role": "user", "parts": [photo_part, text_part]}]
        
        # Определяем провайдера и модель на основе active_mode
        mode = user_mode.get(user_id, "default")
        if mode == "claude":
            provider = "openrouter"
            model = "anthropic/claude-opus-4.8"
        else:
            if os.environ.get("OPENROUTER_API_KEY"):
                provider = "openrouter"
                model = "google/gemini-2.5-flash"
            else:
                provider = "gemini"
                model = MODEL

        logger.info("Анализ фото пользователем %d с помощью %s (%s)", user_id, model, provider)

        if provider in ("openrouter", "omniroute"):
            # Для внешних провайдеров преобразуем payload в формат OpenAI (с base64 image_url) и стримим
            stream_iter = await call_external_llm(
                provider=provider,
                model=model,
                history=current_request,
                system_instruction=SYSTEM_PROMPT_IMAGE,
                stream=True,
                max_tokens=8192
            )
            reply_text = await native_stream_reply(update, context.bot.token, stream_iter)
        else:
            # Для официального Gemini
            stream_config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_IMAGE,
                max_output_tokens=8192,
            )
            stream_iter = gemini_stream(contents=current_request, config=stream_config)
            reply_text = await native_stream_reply(update, context.bot.token, stream_iter)

    except Exception as exc:
        logger.error("Ошибка при обработке изображения (%s): %s", model if 'model' in locals() else 'unknown', exc)
        await update.message.reply_text("⚠️ Не удалось обработать изображение. Попробуй ещё раз.")
        return
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Сохраняем в историю диалога
    add_message(user_id, "user",  [{"text": f"[Изображение] {user_prompt}"}])
    if reply_text:
        add_message(user_id, "model", [{"text": reply_text}])

# ---------------------------------------------------------------------------
# Текстовые сообщения в обычном чате
# ---------------------------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text or ""

    # Проверяем состояние ожидания ввода ( think / image / scheduler )
    state = user_state.pop(user_id, None)
    if state == "awaiting_schedule_chat":
        chat_id = None
        if update.message.forward_from_chat:
            chat_id = update.message.forward_from_chat.id
        elif update.message.forward_from:
            chat_id = update.message.forward_from.id
        else:
            text_input = user_text.strip()
            if text_input.startswith("@"):
                chat_id = text_input
            else:
                try:
                    chat_id = int(text_input)
                except ValueError:
                    user_state[user_id] = state
                    await update.message.reply_text("❌ Неверный формат получателя. Пожалуйста, отправьте числовой ID или юзернейм.")
                    return

        draft = user_schedule_drafts.setdefault(user_id, {})
        draft["chat_id"] = chat_id

        user_state[user_id] = "awaiting_schedule_time"
        now_str = datetime.now().strftime("%d.%m.%Y %H:%M")
        await update.message.reply_text(
            f"📅 <b>Новое отложенное сообщение (Шаг 2 из 3)</b>\n\n"
            f"Получатель: <code>{chat_id}</code>\n\n"
            f"Теперь укажите <b>дату и время отправки</b> в формате: <code>ДД.ММ.ГГГГ ЧЧ:ММ</code>\n"
            f"Пример: <code>24.06.2026 15:30</code>\n\n"
            f"Текущее время сервера: <b>{now_str}</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Отмена", callback_data="settings:scheduler:menu")]])
        )
        return

    elif state == "awaiting_schedule_time":
        time_text = user_text.strip()
        try:
            dt = datetime.strptime(time_text, "%d.%m.%Y %H:%M")
        except ValueError:
            user_state[user_id] = state
            await update.message.reply_text(
                "❌ Неверный формат времени. Пожалуйста, введите дату и время в формате <code>ДД.ММ.ГГГГ ЧЧ:ММ</code> (например, 24.06.2026 15:30).",
                parse_mode=ParseMode.HTML
            )
            return

        now = datetime.now()
        if dt <= now:
            user_state[user_id] = state
            await update.message.reply_text("❌ Время отправки должно быть в будущем. Попробуйте еще раз:")
            return

        draft = user_schedule_drafts.setdefault(user_id, {})
        draft["run_at"] = dt.isoformat()

        user_state[user_id] = "awaiting_schedule_text"
        await update.message.reply_text(
            f"📅 <b>Новое отложенное сообщение (Шаг 3 из 3)</b>\n\n"
            f"Получатель: <code>{draft['chat_id']}</code>\n"
            f"Время отправки: <b>{dt.strftime('%d.%m.%Y %H:%M')}</b>\n\n"
            f"Теперь напишите <b>текст сообщения</b>, которое нужно отправить:",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Отмена", callback_data="settings:scheduler:menu")]])
        )
        return

    elif state == "awaiting_schedule_text":
        msg_text = user_text
        draft = user_schedule_drafts.pop(user_id, {})

        chat_id = draft.get("chat_id")
        run_at_str = draft.get("run_at")
        dt = datetime.fromisoformat(run_at_str)

        conn_id = None
        for cid, uid in business_connections.items():
            if uid == user_id:
                conn_id = cid
                break

        task_id = f"task_{int(dt.timestamp())}_{random.randint(1000, 9999)}"
        task_data = {
            "id": task_id,
            "chat_id": chat_id,
            "conn_id": conn_id,
            "text": msg_text,
            "run_at": run_at_str,
            "creator_id": user_id
        }

        tasks = _load_scheduled_tasks()
        tasks.append(task_data)
        _save_scheduled_tasks(tasks)

        if context.job_queue:
            context.job_queue.run_once(
                send_scheduled_message_callback,
                when=dt,
                data=task_id,
                name=task_id,
                chat_id=chat_id
            )

        await update.message.reply_text(
            f"✅ <b>Сообщение успешно запланировано!</b>\n\n"
            f"Получатель: <code>{chat_id}</code>\n"
            f"Время: <b>{dt.strftime('%d.%m.%Y %H:%M')}</b>\n"
            f"Текст: {msg_text}",
            parse_mode=ParseMode.HTML
        )

        await send_new_menu(update.effective_chat.id, context, user_id)
        return

    if state == "awaiting_think":
        allowed, reason = check_and_record_premium_request(update.effective_user)
        if not allowed:
            if reason == "need_premium":
                await update.message.reply_text("❌ Функция Глубокого мышления доступна только по Premium подписке! Оформите её с помощью /premium.")
            elif reason == "limit_exceeded":
                await update.message.reply_text("❌ Вы исчерпали лимит 5 запросов Глубокого мышления в неделю!")
            if update.effective_chat.type == "private":
                await send_new_menu(update.effective_chat.id, context, user_id)
            return
            
        if update.effective_chat.type == "private":
            await delete_old_menu(update.effective_chat.id, context)
        await handle_think_logic(update, context, user_text)
        if update.effective_chat.type == "private":
            await send_new_menu(update.effective_chat.id, context, user_id)
        return
    elif state == "awaiting_image":
        if update.effective_chat.type == "private":
            await delete_old_menu(update.effective_chat.id, context)
        await handle_image_logic(update, context, user_text)
        if update.effective_chat.type == "private":
            await send_new_menu(update.effective_chat.id, context, user_id)
        return

    chat_type = update.effective_chat.type
    mode = user_mode.get(user_id, "default")

    if mode == "claude":
        allowed, reason = check_and_record_premium_request(update.effective_user)
        if not allowed:
            if reason == "need_premium":
                await update.message.reply_text("❌ Режим Claude доступен только по Premium подписке! Переключаю вас на Gemini.")
            elif reason == "limit_exceeded":
                await update.message.reply_text("❌ Вы исчерпали лимит 5 премиум-запросов в неделю! Переключаю вас на Gemini.")
            set_user_mode(user_id, "default")
            mode = "default"

    if chat_type in ("group", "supergroup"):
        bot_username = (context.bot.username or "").lower()
        msg = update.message

        mentioned = any(
            msg.text[e.offset: e.offset + e.length].lstrip("@").lower() == bot_username
            for e in (msg.entities or [])
            if e.type == "mention"
        )
        replying_to_bot = bool(
            msg.reply_to_message
            and msg.reply_to_message.from_user
            and (msg.reply_to_message.from_user.username or "").lower() == bot_username
        )

        if not mentioned and not replying_to_bot:
            return

        if bot_username:
            user_text = re.sub(rf"@{re.escape(bot_username)}", "", user_text, flags=re.IGNORECASE).strip()

    add_message(user_id, "user", [{"text": user_text}])
    history = get_history(user_id)

    # Определяем провайдера и модель на основе выбранного режима
    if mode == "claude":
        provider = "openrouter"
        model = "anthropic/claude-opus-4.8"
    else:
        if os.environ.get("OPENROUTER_API_KEY"):
            provider = "openrouter"
            model = "google/gemini-2.5-flash-lite"
        else:
            provider = "gemini"
            model = MODEL

    try:
        if provider in ("openrouter", "omniroute"):
            stream_iter = await call_external_llm(
                provider=provider,
                model=model,
                history=history,
                system_instruction=SYSTEM_PROMPT_DEFAULT,
                stream=True,
                max_tokens=8192
            )
            reply_text = await native_stream_reply(update, context.bot.token, stream_iter)
        else:
            stream_config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_DEFAULT,
                max_output_tokens=8192,
            )
            stream_iter = gemini_stream(contents=history, config=stream_config)
            reply_text = await native_stream_reply(update, context.bot.token, stream_iter)
            
    except Exception as exc:
        logger.error("Ошибка API (%s / %s): %s", provider, model, exc)
        get_history(user_id).pop()
        await update.message.reply_text("⚠️ Не удалось получить ответ от ИИ. Попробуй ещё раз.")
        return

    if reply_text:
        add_message(user_id, "model", [{"text": reply_text}])

# ---------------------------------------------------------------------------
# Инлайн-режим
# ---------------------------------------------------------------------------
INLINE_CACHE_TIME = 0

async def handle_inline(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.inline_query
    user_text = (query.query or "").strip()

    if not user_text:
        await query.answer(
            results=[],
            switch_pm_text="Напиши вопрос после @имени бота",
            switch_pm_parameter="start",
            cache_time=INLINE_CACHE_TIME,
        )
        return

    logger.info("Инлайн-запрос от user=%d: %s", query.from_user.id, user_text[:60])
    user = query.from_user
    user_id = user.id
    mode = user_mode.get(user_id, "default")
    if mode == "claude":
        allowed, reason = check_and_record_premium_request(user)
        if not allowed:
            set_user_mode(user_id, "default")
            mode = "default"

    try:
        if mode == "claude":
            answer_raw = await call_external_llm(
                provider="openrouter",
                model="anthropic/claude-opus-4.8",
                history=[{"role": "user", "parts": [{"text": user_text}]}],
                system_instruction=SYSTEM_PROMPT_DEFAULT,
                stream=False,
                max_tokens=2048
            )
        else:
            if os.environ.get("OPENROUTER_API_KEY"):
                answer_raw = await call_external_llm(
                    provider="openrouter",
                    model="google/gemini-2.5-flash-lite",
                    history=[{"role": "user", "parts": [{"text": user_text}]}],
                    system_instruction=SYSTEM_PROMPT_DEFAULT,
                    stream=False,
                    max_tokens=2048
                )
            else:
                response = gemini_generate(
                    contents=[{"role": "user", "parts": [{"text": user_text}]}],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT_DEFAULT,
                        max_output_tokens=2048,
                    ),
                )
                answer_raw = response.text or "Не удалось получить ответ."
    except Exception as exc:
        logger.error("Ошибка в инлайн-режиме: %s", exc)
        answer_raw = "⚠️ Ошибка при обращении к ИИ. Попробуй ещё раз."

    answer_html = md_to_html(answer_raw)
    short_preview = answer_raw[:120].replace("\n", " ") + ("…" if len(answer_raw) > 120 else "")

    import hashlib
    result_id = hashlib.md5(f"{query.from_user.id}:{user_text}".encode()).hexdigest()[:16]

    result = InlineQueryResultArticle(
        id=result_id,
        title=f"💬 {user_text[:50]}",
        description=short_preview,
        input_message_content=InputTextMessageContent(
            message_text=answer_html,
            parse_mode=ParseMode.HTML,
        ),
    )

    await query.answer(
        results=[result],
        cache_time=INLINE_CACHE_TIME,
        is_personal=True,
    )

# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
async def send_scheduled_message_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    job = context.job
    task_id = job.data

    tasks = _load_scheduled_tasks()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        logger.warning("Отложенная задача %s не найдена в базе!", task_id)
        return

    chat_id = task["chat_id"]
    conn_id = task.get("conn_id")
    text = task["text"]

    logger.info("Выполнение отложенной отправки %s в чат %s", task_id, chat_id)

    formatted = md_to_html(text)
    parts = split_message(formatted)
    for part in parts:
        kwargs = {"parse_mode": ParseMode.HTML}
        if conn_id:
            kwargs["business_connection_id"] = conn_id
        try:
            target_chat = chat_id
            if isinstance(chat_id, str) and not chat_id.startswith("@"):
                try:
                    target_chat = int(chat_id)
                except ValueError:
                    pass
            await context.bot.send_message(chat_id=target_chat, text=part, **kwargs)
        except Exception as e:
            logger.error("Ошибка отправки отложенного сообщения в %s: %s", chat_id, e)

    # Удаляем выполненную задачу
    remaining_tasks = [t for t in tasks if t["id"] != task_id]
    _save_scheduled_tasks(remaining_tasks)

async def post_init(app) -> None:
    """Устанавливает меню команд бота после запуска."""
    from telegram import BotCommand
    await app.bot.set_my_commands([
        BotCommand("start",    "👋 Главное меню бота"),
        BotCommand("gemini",   "🤖 Режим Gemini (Бесплатно)"),
        BotCommand("claude",   "👑 Режим Claude Opus (Премиум)"),
        BotCommand("think",    "🧠 Глубокое мышление"),
        BotCommand("image",    "🎨 Генерация картинок"),
        BotCommand("history",  "📜 Управление историей"),
        BotCommand("settings", "⚙️ Настройки бота"),
        BotCommand("schedule", "📅 Отложенное сообщение"),
        BotCommand("reset",    "🗑️ Очистить историю"),
    ])
    logger.info("Меню команд обновлено.")

    # Восстановление отложенных сообщений из базы данных
    if app.job_queue:
        tasks = _load_scheduled_tasks()
        now = datetime.now()
        restored_count = 0
        for task in tasks:
            try:
                run_at = datetime.fromisoformat(task["run_at"])
                if run_at > now:
                    app.job_queue.run_once(
                        send_scheduled_message_callback,
                        when=run_at,
                        data=task["id"],
                        name=task["id"],
                        chat_id=task["chat_id"]
                    )
                    restored_count += 1
                else:
                    logger.info("Отложенное сообщение %s пропущено (время в прошлом)", task["id"])
            except Exception as e:
                logger.error("Ошибка при восстановлении задачи %s: %s", task.get("id"), e)
        logger.info("Восстановлено отложенных сообщений: %d", restored_count)


def main() -> None:
    logger.info("Запуск Telegram-бота...")
    _load_state()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).job_queue().build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("reset",    cmd_reset))
    app.add_handler(CommandHandler("history",  cmd_history))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("persona",  cmd_persona))
    app.add_handler(CommandHandler("claude",   cmd_claude))
    app.add_handler(CommandHandler("gemini",   cmd_gemini))
    app.add_handler(CommandHandler("think",    cmd_think))
    app.add_handler(CommandHandler("image",    cmd_image))
    app.add_handler(CommandHandler("generate", cmd_image))
    app.add_handler(CommandHandler("automate", cmd_automate))
    app.add_handler(CommandHandler("premium",  cmd_premium))
    app.add_handler(CommandHandler("grant_premium", cmd_grant_premium))
    app.add_handler(CommandHandler("revoke_premium", cmd_revoke_premium))
    app.add_handler(CommandHandler("list_premium", cmd_list_premium))
    app.add_handler(CommandHandler("schedule", cmd_schedule))

    # Inline-кнопки
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Инлайн-режим
    app.add_handler(InlineQueryHandler(handle_inline))

    # Business Bot
    app.add_handler(BusinessConnectionHandler(handle_business_connection))
    app.add_handler(MessageHandler(filters.UpdateType.BUSINESS_MESSAGE, handle_business_message))

    # Изображения
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_photo))

    # Текст
    _group_mention_filter = (
        filters.ChatType.GROUPS & (filters.Entity("mention") | filters.REPLY)
    )
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & (filters.ChatType.PRIVATE | _group_mention_filter),
        handle_message,
    ))

    logger.info("Бот запущен.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
