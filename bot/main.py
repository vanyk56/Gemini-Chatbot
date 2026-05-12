"""
Telegram-бот с ИИ на базе Google Gemini.

Секреты:
  - TELEGRAM_BOT_TOKEN       — токен бота от @BotFather
  - GEMINI_API_KEY           — ключ Google AI Studio
  - AI_INTEGRATIONS_GEMINI_BASE_URL / AI_INTEGRATIONS_GEMINI_API_KEY (Replit proxy)

Команды:
  /start    — приветствие
  /reset    — очистить историю диалога
  /history  — просмотр и управление историей
  /settings — настройки бота
  /OBZR     — режим учебника ОБЖ 8–9 кл.
  /exit     — выйти из режима учебника
"""

import os
import re
import json
import time
import asyncio
import logging
import tempfile
import base64
from pathlib import Path
from datetime import datetime

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
# Токены / клиент
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

_REPLIT_BASE_URL = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")
_REPLIT_API_KEY  = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY")
_DIRECT_API_KEY  = os.environ.get("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise EnvironmentError("TELEGRAM_BOT_TOKEN не задан!")

if _REPLIT_BASE_URL and _REPLIT_API_KEY:
    client = genai.Client(
        api_key=_REPLIT_API_KEY,
        http_options=types.HttpOptions(base_url=_REPLIT_BASE_URL, api_version=""),
    )
elif _DIRECT_API_KEY:
    client = genai.Client(api_key=_DIRECT_API_KEY)
else:
    raise EnvironmentError(
        "Нужен GEMINI_API_KEY или AI_INTEGRATIONS_GEMINI_BASE_URL + AI_INTEGRATIONS_GEMINI_API_KEY"
    )

MODEL        = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
PAGES_DIR    = Path("bot/books/pages")
BOOK_PATH    = Path("bot/books/obzr.txt")
PROGRESS_PATH = Path("bot/books/progress.txt")
MAX_MSG_LEN  = 4000
MAX_HISTORY  = 50   # максимум сообщений в памяти

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
    text = re.sub(r"\\[a-zA-Z]+", "", text)
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
# Нативный стриминг через Telegram Bot API sendMessageDraft
# ---------------------------------------------------------------------------
_draft_counter = 0

def _next_draft_id() -> int:
    global _draft_counter
    _draft_counter = (_draft_counter % 2_000_000_000) + 1
    return _draft_counter

async def _send_draft(token: str, chat_id: int | str, draft_id: int, text: str) -> None:
    """Вызывает sendMessageDraft через raw Bot API (метод не поддерживается PTB 22.7)."""
    import httpx
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
    """
    Нативный стриминг:
      1. sendMessageDraft → анимированный черновик (Thinking... / текст по мере генерации)
      2. sendMessage      → финальное сообщение (черновик исчезает сам)
    Возвращает полный сырой текст.
    """
    chat_id  = update.effective_chat.id
    draft_id = _next_draft_id()

    # Показываем "Thinking..." (пустой text = стандартный плейсхолдер Telegram)
    await _send_draft(token, chat_id, draft_id, "")

    queue: asyncio.Queue = asyncio.Queue()

    def _produce() -> None:
        try:
            for chunk in stream_iter:
                txt = getattr(chunk, "text", None) or ""
                if txt:
                    queue.put_nowait(txt)
        finally:
            queue.put_nowait(None)

    asyncio.get_running_loop().run_in_executor(None, _produce)

    accumulated = ""
    last_update  = 0.0

    while True:
        try:
            chunk_text = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            break
        if chunk_text is None:
            break
        accumulated += chunk_text

        now = time.monotonic()
        if now - last_update >= DRAFT_UPDATE_INTERVAL and len(accumulated) >= DRAFT_MIN_CHARS:
            await _send_draft(token, chat_id, draft_id, latex_to_text(accumulated))
            last_update = now

    # Финальное сообщение — персистентное; черновик исчезает автоматически
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

conversation_history: dict[int, list]          = {}
conversation_timestamps: dict[int, list[str]]  = {}
user_mode: dict[int, str]                       = {}
user_settings: dict[int, dict]                  = {}
business_connections: dict[str, int]            = {}  # conn_id → user_id

def _load_state() -> None:
    global user_mode, user_settings, business_connections
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
        "max_history": 20,
        "language": "auto",
    })

def set_user_mode(user_id: int, mode: str) -> None:
    user_mode[user_id] = mode
    _save_modes()

def get_history(user_id: int) -> list:
    return conversation_history.setdefault(user_id, [])

def get_timestamps(user_id: int) -> list:
    return conversation_timestamps.setdefault(user_id, [])

def add_message(user_id: int, role: str, parts_data: list) -> None:
    settings = get_settings(user_id)
    max_h = settings.get("max_history", 20)
    history = get_history(user_id)
    timestamps = get_timestamps(user_id)
    history.append({"role": role, "parts": parts_data})
    timestamps.append(datetime.now().strftime("%H:%M"))
    # Обрезаем историю если превышен лимит (попарно user+model)
    while len(history) > max_h * 2:
        history.pop(0)
        if timestamps:
            timestamps.pop(0)

def clear_user(user_id: int) -> None:
    conversation_history.pop(user_id, None)
    conversation_timestamps.pop(user_id, None)
    user_mode.pop(user_id, None)
    _save_modes()

# ---------------------------------------------------------------------------
# Учебник ОБЖ
# ---------------------------------------------------------------------------
def load_book_text() -> str:
    if BOOK_PATH.exists():
        text = BOOK_PATH.read_text(encoding="utf-8").strip()
        if text:
            return text
    return ""

def get_pages_processed() -> int:
    if PROGRESS_PATH.exists():
        try:
            return int(PROGRESS_PATH.read_text().strip())
        except ValueError:
            return 0
    return 0

_book_pages: dict[int, str] = {}

def load_book_pages() -> dict[int, str]:
    global _book_pages
    if _book_pages:
        return _book_pages
    if not BOOK_PATH.exists():
        return {}
    raw = BOOK_PATH.read_text(encoding="utf-8")
    for m in re.finditer(r"=== Страница (\d+) ===\n(.*?)(?==== Страница \d+ ===|\Z)", raw, re.DOTALL):
        _book_pages[int(m.group(1))] = m.group(2).strip()
    logger.info("Загружено страниц из кэша: %d", len(_book_pages))
    return _book_pages

_STOP_WORDS = {
    "что","как","это","так","еще","ещё","уже","вот","или","при","для","все",
    "того","этот","эта","эти","этого","этой","такой","такое","такая","такие",
    "были","была","было","быть","есть","нет","если","можно","нужно","надо",
    "который","которые","которая","которого","которой","которую","которым",
    "чтобы","когда","тогда","после","более","очень","какой","какие","между",
    "также","самый","самая","самое","самые","него","нему","ними","них",
    "своей","своего","своих","своим","свою","свои","меня","тебя","себя",
    "мне","тебе","себе","они","она","оно","нас","вас","про",
}

def search_relevant_pages(question: str, pages: dict[int, str], top_n: int = 4) -> list[int]:
    all_words = [w.lower().strip("?!.,;:«»\"'()") for w in question.split()]
    words = [w for w in all_words if len(w) >= 4 and w not in _STOP_WORDS]
    if not words:
        words = [w for w in all_words if len(w) >= 3 and w not in _STOP_WORDS]
    if not words:
        return []
    scores: dict[int, float] = {}
    for page_num, text in pages.items():
        score = sum(text.lower().count(w) * (len(w) ** 2) for w in words)
        if score > 0:
            scores[page_num] = score
    return sorted(scores, key=lambda p: scores[p], reverse=True)[:top_n]

def build_obzr_prompt(relevant_pages: dict[int, str]) -> str:
    pages_text = "\n\n".join(f"[Стр. {n}]\n{t}" for n, t in sorted(relevant_pages.items()))
    return (
        "Ты — строгий ассистент по ОБЖ (Основы безопасности жизнедеятельности).\n"
        "Отвечай ИСКЛЮЧИТЕЛЬНО по тексту страниц учебника ниже.\n"
        "НЕ добавляй факты из своих знаний. Ответ — максимум 150 слов.\n\n"
        "=== СТРАНИЦЫ УЧЕБНИКА ===\n" + pages_text + "\n=== КОНЕЦ ==="
    )

def get_page_image(page_num: int) -> Path | None:
    path = PAGES_DIR / f"page_{page_num:03d}.jpg"
    return path if path.exists() else None

# ---------------------------------------------------------------------------
# Системные промпты
# ---------------------------------------------------------------------------
_NO_LATEX = (
    "ВАЖНО: Telegram не поддерживает LaTeX. НЕ используй $, \\frac, \\times, \\sqrt и т.д. "
    "Дроби пиши как '3/4', умножение как '×', корень как '√', степень как '^2'. "
    "Используй Unicode: ×, ÷, ±, ≤, ≥, √, π."
)

SYSTEM_PROMPT_DEFAULT = (
    "Ты — дружелюбный и умный ИИ-собеседник в Telegram. "
    "Отвечай развёрнуто, но по делу. Используй эмодзи умеренно. "
    "Пиши на том же языке, на котором пишет пользователь. " + _NO_LATEX
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

# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(
        f"Привет, {user.first_name}! 👋\n"
        "Я ИИ-собеседник на базе Google Gemini.\n\n"
        "🖼 Умею работать с <b>изображениями</b> — решу задачи, объясню текст на фото.\n"
        "💼 Поддерживаю <b>автоматизацию чатов</b> — подключи меня в настройках Telegram Business.\n\n"
        "📋 <b>Команды:</b>\n"
        "/history  — история диалога и управление ей\n"
        "/settings — настройки бота\n"
        "/persona  — настроить личность для авто-ответчика\n"
        "/OBZR     — режим учебника ОБЖ 8–9 кл.\n"
        "/reset    — очистить историю\n"
        "/exit     — выйти из режима учебника\n\n"
        "Напиши что-нибудь или отправь фото!",
        parse_mode=ParseMode.HTML,
    )

# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------
async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_user(update.effective_user.id)
    await update.message.reply_text("🗑️ История очищена. Начинаем с чистого листа!")

# ---------------------------------------------------------------------------
# /history — просмотр истории с кнопками управления
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

    # Последние 3 сообщения для превью
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
# /settings — настройки бота
# ---------------------------------------------------------------------------
async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    settings = get_settings(user_id)
    mode = user_mode.get(user_id, "default")
    auto = settings.get("auto_reply", True)
    max_h = settings.get("max_history", 20)
    persona = settings.get("persona", "")

    mode_label = {"default": "💬 Обычный", "obzr": "📖 ОБЖ"}.get(mode, mode)
    auto_label = "✅ Вкл" if auto else "❌ Выкл"
    persona_label = (persona[:40] + "...") if len(persona) > 40 else (persona or "не задана")

    text = (
        "⚙️ <b>Настройки бота</b>\n\n"
        f"🗂 Режим: <b>{mode_label}</b>\n"
        f"🔄 Авто-ответ в бизнес-чатах: <b>{auto_label}</b>\n"
        f"📝 Лимит истории: <b>{max_h} сообщений</b>\n"
        f"🎭 Личность авто-ответчика: <i>{persona_label}</i>\n\n"
        "Задай личность через /persona — бот будет отвечать естественнее."
    )

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("💬 Обычный режим", callback_data="mode:default"),
            InlineKeyboardButton("📖 Режим ОБЖ",     callback_data="mode:obzr"),
        ],
        [
            InlineKeyboardButton(
                f"🔄 Авто-ответ: {'✅' if auto else '❌'}",
                callback_data="settings:auto_reply:toggle",
            ),
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
        [InlineKeyboardButton("🗑️ Очистить историю", callback_data="history:clear")],
        [InlineKeyboardButton("❌ Закрыть", callback_data="menu:close")],
    ])

    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

# ---------------------------------------------------------------------------
# /persona — задать личность для авто-ответчика
# ---------------------------------------------------------------------------
async def cmd_persona(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    settings = get_settings(user_id)
    persona = settings.get("persona", "")

    args = context.args
    if args:
        # /persona Меня зовут Дима, мне 20 лет, занимаюсь бизнесом...
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
            "Напиши после команды информацию о себе — бот будет отвечать естественнее:\n\n"
            "<code>/persona Меня зовут Артём, мне 19 лет. Учусь в универе, "
            "увлекаюсь музыкой и спортом. Общаюсь неформально.</code>\n\n"
            "Чем больше деталей — тем естественнее ответы.",
            parse_mode=ParseMode.HTML,
        )

# ---------------------------------------------------------------------------
# /OBZR
# ---------------------------------------------------------------------------
async def cmd_obzr(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not load_book_text():
        await update.message.reply_text("⏳ Учебник ещё обрабатывается. Попробуй через несколько минут.")
        return
    conversation_history.pop(user_id, None)
    set_user_mode(user_id, "obzr")
    await update.message.reply_text(
        "📖 <b>Режим учебника ОБЖ активирован!</b>\n"
        f"✅ Загружено страниц: {get_pages_processed()}/239\n\n"
        "Задавай вопросы — отвечаю строго по учебнику.\n"
        "/exit — вернуться к обычному режиму",
        parse_mode=ParseMode.HTML,
    )

# ---------------------------------------------------------------------------
# /exit
# ---------------------------------------------------------------------------
async def cmd_exit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_mode.get(user_id) == "obzr":
        clear_user(user_id)
        await update.message.reply_text("✅ Вышел из режима учебника. Теперь я обычный ИИ-собеседник.")
    else:
        await update.message.reply_text("Ты и так в обычном режиме.")

# ---------------------------------------------------------------------------
# Callback-кнопки (история / настройки / режим)
# ---------------------------------------------------------------------------
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data

    if data == "menu:close":
        await query.message.delete()
        return

    if data == "history:clear":
        clear_user(user_id)
        await query.edit_message_text("🗑️ История диалога очищена!")
        return

    if data == "history:keep5":
        history = get_history(user_id)
        timestamps = get_timestamps(user_id)
        if len(history) > 10:
            conversation_history[user_id] = history[-10:]
            conversation_timestamps[user_id] = timestamps[-10:]
        await query.edit_message_text("✂️ Оставлены последние 5 обменов.")
        return

    if data.startswith("mode:"):
        new_mode = data.split(":")[1]
        if new_mode == "obzr" and not load_book_text():
            await query.answer("⏳ Учебник ещё обрабатывается!", show_alert=True)
            return
        conversation_history.pop(user_id, None)
        set_user_mode(user_id, new_mode)
        label = {"default": "💬 обычный", "obzr": "📖 ОБЖ"}.get(new_mode, new_mode)
        await query.edit_message_text(f"✅ Режим изменён на {label}.")
        return

    if data.startswith("settings:"):
        parts = data.split(":")
        key = parts[1]
        val = parts[2] if len(parts) > 2 else None
        settings = get_settings(user_id)

        if key == "auto_reply" and val == "toggle":
            settings["auto_reply"] = not settings.get("auto_reply", True)
            _save_settings()
            status = "включён" if settings["auto_reply"] else "выключен"
            await query.edit_message_text(f"🔄 Авто-ответ в бизнес-чатах {status}.")
            return

        if key == "max_history" and val:
            settings["max_history"] = int(val)
            _save_settings()
            await query.edit_message_text(f"📝 Лимит истории установлен: {val} сообщений.")
            return

    if data.startswith("persona:"):
        action = data.split(":")[1]
        settings = get_settings(user_id)
        if action == "reset":
            settings.pop("persona", None)
            _save_settings()
            await query.edit_message_text("🗑️ Личность авто-ответчика сброшена.")
        elif action == "prompt":
            await query.edit_message_text(
                "🎭 Отправь команду с описанием себя:\n\n"
                "<code>/persona Меня зовут Артём, мне 19 лет. Учусь в универе, "
                "увлекаюсь музыкой и спортом. Общаюсь неформально.</code>",
                parse_mode=ParseMode.HTML,
            )
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
                    "🔄 Авто-ответ можно отключить там же."
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

    # Если подключение не в памяти (бот перезапустился) — запросим у Telegram API
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

    settings = get_settings(owner_id)
    if not settings.get("auto_reply", True):
        logger.info("Авто-ответ выключен для user=%d", owner_id)
        return

    # Не отвечаем на сообщения самого владельца
    if msg.from_user and msg.from_user.id == owner_id:
        return

    sender_name = msg.from_user.first_name if msg.from_user else "Собеседник"
    logger.info("Business сообщение от %s (conn=%s): %s", sender_name, conn_id, msg.text[:50])

    await context.bot.send_chat_action(
        chat_id=msg.chat.id,
        action="typing",
        business_connection_id=conn_id,
    )

    # Используем историю владельца для контекста business-чата
    key = f"biz_{conn_id}_{msg.chat.id}"
    biz_history = conversation_history.setdefault(key, [])
    biz_history.append({"role": "user", "parts": [{"text": msg.text}]})

    try:
        persona = get_settings(owner_id).get("persona", "")
        response = client.models.generate_content(
            model=MODEL,
            contents=biz_history,
            config=types.GenerateContentConfig(
                system_instruction=build_business_prompt(persona),
                max_output_tokens=1024,
            ),
        )
        reply_text = response.text
    except Exception as exc:
        logger.error("Ошибка Gemini (business): %s", exc)
        return

    biz_history.append({"role": "model", "parts": [{"text": reply_text}]})
    # Обрезаем историю бизнес-чата
    if len(biz_history) > 40:
        conversation_history[key] = biz_history[-40:]

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
# Обработка изображений
# ---------------------------------------------------------------------------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
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
        with open(tmp_path, "rb") as f:
            image_bytes = f.read()

        history = get_history(user_id)
        response = client.models.generate_content(
            model=MODEL,
            contents=history + [{"role": "user", "parts": [
                {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}},
                {"text": user_prompt},
            ]}],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_IMAGE,
                max_output_tokens=8192,
            ),
        )
        reply_text = response.text

    except Exception as exc:
        logger.error("Ошибка Gemini API (изображение): %s", exc)
        await update.message.reply_text("⚠️ Не удалось обработать изображение. Попробуй ещё раз.")
        return
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    add_message(user_id, "user",  [{"text": f"[Изображение] {user_prompt}"}])
    add_message(user_id, "model", [{"text": reply_text}])
    await send_reply(update, reply_text)

# ---------------------------------------------------------------------------
# Текстовые сообщения
# ---------------------------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text or ""
    chat_type = update.effective_chat.type
    mode = user_mode.get(user_id, "default")

    # В группах — отвечаем только при упоминании @бота или ответе на его сообщение
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

        # Убираем @упоминание из текста перед отправкой в Gemini
        if bot_username:
            user_text = re.sub(rf"@{re.escape(bot_username)}", "", user_text, flags=re.IGNORECASE).strip()

    relevant_page_nums: list[int] = []

    if mode == "obzr":
        pages = load_book_pages()
        if not pages:
            await update.message.reply_text("⏳ Учебник ещё обрабатывается. Попробуй позже.")
            return
        relevant_page_nums = search_relevant_pages(user_text, pages, top_n=4)
        if relevant_page_nums:
            system_prompt = build_obzr_prompt({n: pages[n] for n in relevant_page_nums if n in pages})
        else:
            await update.message.reply_text(
                "🔍 По вашему вопросу ничего не найдено в учебнике ОБЖ.\n"
                "Попробуй перефразировать или уточнить тему."
            )
            return
    else:
        system_prompt = SYSTEM_PROMPT_DEFAULT

    add_message(user_id, "user", [{"text": user_text}])
    history = get_history(user_id)

    try:
        stream_iter = client.models.generate_content_stream(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=8192,
            ),
        )
        reply_text = await native_stream_reply(
            update,
            context.bot.token,
            stream_iter,
        )
    except Exception as exc:
        logger.error("Ошибка Gemini API: %s", exc)
        get_history(user_id).pop()
        await update.message.reply_text("⚠️ Не удалось получить ответ от ИИ. Попробуй ещё раз.")
        return

    if reply_text:
        add_message(user_id, "model", [{"text": reply_text}])

    if mode == "obzr" and relevant_page_nums:
        img_path = get_page_image(relevant_page_nums[0])
        if img_path:
            try:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
                with open(img_path, "rb") as f:
                    await update.message.reply_photo(
                        photo=f,
                        caption=f"📄 Страница {relevant_page_nums[0]} учебника ОБЖ",
                    )
            except Exception as e:
                logger.warning("Не удалось отправить страницу: %s", e)

# ---------------------------------------------------------------------------
# Инлайн-режим: @бот вопрос — в любом чате без добавления бота
# ---------------------------------------------------------------------------
INLINE_CACHE_TIME = 0  # секунд (0 = не кешировать, у каждого свой ответ)

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

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[{"role": "user", "parts": [{"text": user_text}]}],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_DEFAULT,
                max_output_tokens=2048,
            ),
        )
        answer_raw = response.text or "Не удалось получить ответ."
    except Exception as exc:
        logger.error("Ошибка Gemini (inline): %s", exc)
        answer_raw = "⚠️ Ошибка при обращении к ИИ. Попробуй ещё раз."

    answer_html = md_to_html(answer_raw)
    # Telegram inline: максимум 4096 символов в тексте результата
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
async def post_init(app) -> None:
    """Устанавливает меню команд бота после запуска."""
    from telegram import BotCommand
    await app.bot.set_my_commands([
        BotCommand("start",    "👋 Приветствие и список команд"),
        BotCommand("history",  "📜 История диалога и управление ей"),
        BotCommand("settings", "⚙️ Настройки бота"),
        BotCommand("persona",  "🎭 Личность авто-ответчика"),
        BotCommand("reset",    "🗑️ Очистить историю диалога"),
        BotCommand("obzr",     "📖 Режим учебника ОБЖ 8–9 кл."),
        BotCommand("exit",     "🚪 Выйти из режима учебника"),
    ])
    logger.info("Меню команд обновлено.")


def main() -> None:
    logger.info("Запуск Telegram-бота...")
    _load_state()
    load_book_pages()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("reset",    cmd_reset))
    app.add_handler(CommandHandler("history",  cmd_history))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("persona",  cmd_persona))
    app.add_handler(CommandHandler("OBZR",     cmd_obzr))
    app.add_handler(CommandHandler("obzr",     cmd_obzr))
    app.add_handler(CommandHandler("exit",     cmd_exit))

    # Inline-кнопки
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Инлайн-режим (без добавления бота в чат)
    app.add_handler(InlineQueryHandler(handle_inline))

    # Business Bot
    app.add_handler(BusinessConnectionHandler(handle_business_connection))
    app.add_handler(MessageHandler(filters.UpdateType.BUSINESS_MESSAGE, handle_business_message))

    # Изображения
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_photo))

    # Текст: личка — всегда; группы — при упоминании или ответе на бота
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
