"""
Telegram-бот с ИИ на базе Google Gemini (через Replit AI Integrations).

Секреты (Replit Secrets):
  - TELEGRAM_BOT_TOKEN       — токен бота от @BotFather
  - AI_INTEGRATIONS_GEMINI_BASE_URL / AI_INTEGRATIONS_GEMINI_API_KEY
    настраиваются автоматически через Replit Gemini Integration

Команды:
  /start  — приветствие
  /reset  — очистить историю диалога
  /OBZR   — включить режим строгих ответов по учебнику ОБЖ 8–9 кл.
  /exit   — выйти из режима учебника
"""

import os
import re
import json
import logging
import tempfile
from pathlib import Path

from google import genai
from google.genai import types
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
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
# Токены
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

_REPLIT_BASE_URL = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")
_REPLIT_API_KEY  = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY")
_DIRECT_API_KEY  = os.environ.get("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise EnvironmentError("TELEGRAM_BOT_TOKEN не задан!")

if _REPLIT_BASE_URL and _REPLIT_API_KEY:
    _gemini_api_key  = _REPLIT_API_KEY
    _gemini_base_url = _REPLIT_BASE_URL
    _use_proxy = True
elif _DIRECT_API_KEY:
    _gemini_api_key  = _DIRECT_API_KEY
    _gemini_base_url = None
    _use_proxy = False
else:
    raise EnvironmentError(
        "Нужен GEMINI_API_KEY (Railway) "
        "или AI_INTEGRATIONS_GEMINI_BASE_URL + AI_INTEGRATIONS_GEMINI_API_KEY (Replit)."
    )

# ---------------------------------------------------------------------------
# Gemini клиент
# ---------------------------------------------------------------------------
if _use_proxy:
    client = genai.Client(
        api_key=_gemini_api_key,
        http_options=types.HttpOptions(
            base_url=_gemini_base_url,
            api_version="",
        ),
    )
else:
    client = genai.Client(api_key=_gemini_api_key)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
PAGES_DIR = Path("bot/books/pages")
BOOK_PATH = Path("bot/books/obzr.txt")
PROGRESS_PATH = Path("bot/books/progress.txt")
MAX_MSG_LEN = 4000

# ---------------------------------------------------------------------------
# Форматирование: Markdown → HTML для Telegram
# ---------------------------------------------------------------------------
def md_to_html(text: str) -> str:
    text = re.sub(r"^#{1,3}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text

def split_message(text: str, max_len: int = MAX_MSG_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts = []
    current = ""
    for paragraph in text.split("\n\n"):
        if len(current) + len(paragraph) + 2 <= max_len:
            current += ("" if not current else "\n\n") + paragraph
        else:
            if current:
                parts.append(current)
            if len(paragraph) > max_len:
                lines = paragraph.split("\n")
                for line in lines:
                    if len(current) + len(line) + 1 <= max_len:
                        current += ("" if not current else "\n") + line
                    else:
                        if current:
                            parts.append(current)
                        current = line
            else:
                current = paragraph
    if current:
        parts.append(current)
    return parts or [text[:max_len]]

# ---------------------------------------------------------------------------
# Вспомогательные функции учебника
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
    "что", "как", "это", "так", "еще", "ещё", "уже", "вот", "или", "при", "для", "все",
    "того", "этот", "эта", "эти", "этого", "этой", "такой", "такое", "такая", "такие",
    "были", "была", "было", "быть", "есть", "нет", "если", "можно", "нужно", "надо",
    "который", "которые", "которая", "которого", "которой", "которую", "которым",
    "чтобы", "когда", "тогда", "после", "более", "очень", "какой", "какие", "между",
    "также", "самый", "самая", "самое", "самые", "него", "нему", "ними", "них",
    "своей", "своего", "своих", "своим", "свою", "свои", "меня", "тебя", "себя",
    "мне", "тебе", "себе", "они", "она", "оно", "они", "нас", "вас", "про",
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
        text_lower = text.lower()
        score = sum(text_lower.count(w) * (len(w) ** 2) for w in words)
        if score > 0:
            scores[page_num] = score
    ranked = sorted(scores, key=lambda p: scores[p], reverse=True)
    return ranked[:top_n]

def build_obzr_prompt(relevant_pages: dict[int, str]) -> str:
    pages_text = "\n\n".join(
        f"[Стр. {num}]\n{text}" for num, text in sorted(relevant_pages.items())
    )
    return (
        "Ты — строгий ассистент по ОБЖ (Основы безопасности жизнедеятельности).\n"
        "Ниже приведены КОНКРЕТНЫЕ СТРАНИЦЫ учебника ОБЖ 8–9 класс.\n"
        "Отвечай ИСКЛЮЧИТЕЛЬНО по тексту этих страниц — дословно, без домыслов.\n"
        "НЕ добавляй факты из своих знаний. НЕ используй английские термины если их нет в тексте.\n"
        "Если информации недостаточно — скажи об этом прямо.\n"
        "Ответ должен быть КРАТКИМ — максимум 150 слов. Только суть из учебника.\n"
        "Структура: определение → краткий перечень ключевых пунктов.\n\n"
        "=== ТЕКСТ СТРАНИЦ УЧЕБНИКА ===\n"
        f"{pages_text}\n"
        "=== КОНЕЦ ==="
    )

def get_page_image(page_num: int) -> Path | None:
    path = PAGES_DIR / f"page_{page_num:03d}.jpg"
    return path if path.exists() else None

# ---------------------------------------------------------------------------
# Отправка ответа с HTML и разбивкой
# ---------------------------------------------------------------------------
async def send_reply(update: Update, text: str) -> None:
    formatted = md_to_html(text)
    parts = split_message(formatted)
    for part in parts:
        try:
            await update.message.reply_text(part, parse_mode=ParseMode.HTML)
        except Exception:
            await update.message.reply_text(part)

# ---------------------------------------------------------------------------
# Состояние пользователей
# ---------------------------------------------------------------------------
USER_MODE_FILE = Path("bot/user_modes.json")

conversation_history: dict[int, list] = {}
user_mode: dict[int, str] = {}

def _load_user_modes() -> None:
    global user_mode
    if USER_MODE_FILE.exists():
        try:
            data = json.loads(USER_MODE_FILE.read_text(encoding="utf-8"))
            user_mode = {int(k): v for k, v in data.items()}
            logger.info("Загружены режимы %d пользователей", len(user_mode))
        except Exception as e:
            logger.warning("Не удалось загрузить режимы: %s", e)
            user_mode = {}

def _save_user_modes() -> None:
    try:
        USER_MODE_FILE.write_text(
            json.dumps({str(k): v for k, v in user_mode.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Не удалось сохранить режимы: %s", e)

def set_user_mode(user_id: int, mode: str) -> None:
    user_mode[user_id] = mode
    _save_user_modes()

def get_history(user_id: int) -> list:
    return conversation_history.setdefault(user_id, [])

def add_message(user_id: int, role: str, parts_data: list) -> None:
    get_history(user_id).append({"role": role, "parts": parts_data})

def clear_user(user_id: int) -> None:
    conversation_history.pop(user_id, None)
    user_mode.pop(user_id, None)
    _save_user_modes()

# ---------------------------------------------------------------------------
# Системный промпт обычного режима
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_DEFAULT = (
    "Ты — дружелюбный и умный ИИ-собеседник в Telegram. "
    "Отвечай развёрнуто, но по делу. "
    "Используй эмодзи умеренно. "
    "Пиши на том же языке, на котором пишет пользователь."
)

SYSTEM_PROMPT_IMAGE = (
    "Ты — умный ИИ-ассистент в Telegram. "
    "Пользователь прислал изображение. Внимательно проанализируй его и ответь на вопрос или опиши содержимое. "
    "Если на изображении задача, уравнение или контрольная работа — реши её пошагово. "
    "Если это фото документа или текста — извлеки и объясни содержимое. "
    "Отвечай на том же языке, на котором написан вопрос или подпись к изображению."
)

# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(
        f"Привет, {user.first_name}! 👋\n"
        "Я ИИ-собеседник на базе Google Gemini.\n\n"
        "🖼 <b>Умею работать с изображениями!</b>\n"
        "Просто отправь фото — решу задачи, объясню текст, опишу содержимое.\n\n"
        "📋 <b>Команды:</b>\n"
        "/OBZR — режим строгих ответов по учебнику ОБЖ 8–9 кл.\n"
        "/exit  — выйти из режима учебника\n"
        "/reset — очистить историю диалога\n\n"
        "Просто напиши или отправь фото!",
        parse_mode=ParseMode.HTML,
    )

# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------
async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_user(update.effective_user.id)
    await update.message.reply_text("История очищена. Начинаем с чистого листа! 🗑️")

# ---------------------------------------------------------------------------
# /OBZR
# ---------------------------------------------------------------------------
async def cmd_obzr(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    book_text = load_book_text()
    if not book_text:
        await update.message.reply_text(
            "⏳ Учебник ещё обрабатывается. Попробуй через несколько минут."
        )
        return
    conversation_history.pop(user_id, None)
    set_user_mode(user_id, "obzr")
    pages_done = get_pages_processed()
    await update.message.reply_text(
        "📖 <b>Режим учебника ОБЖ активирован!</b>\n"
        f"✅ Загружено страниц: {pages_done}/239\n\n"
        "Задавай вопросы — отвечаю строго по учебнику.\n"
        "К каждому ответу прикреплю скриншот нужной страницы.\n"
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
        await update.message.reply_text(
            "✅ Вышел из режима учебника. Теперь я снова обычный ИИ-собеседник."
        )
    else:
        await update.message.reply_text("Ты и так в обычном режиме.")

# ---------------------------------------------------------------------------
# Обработка изображений (фото и документы-изображения)
# ---------------------------------------------------------------------------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    # Получаем подпись к фото (если есть) как вопрос пользователя
    caption = update.message.caption or ""
    user_prompt = caption if caption else "Проанализируй это изображение и опиши что на нём."

    # Скачиваем фото (берём наибольшее разрешение)
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
        mime_type = "image/jpeg"
    elif update.message.document and update.message.document.mime_type.startswith("image/"):
        photo_file = await update.message.document.get_file()
        mime_type = update.message.document.mime_type
    else:
        await update.message.reply_text("⚠️ Не удалось получить изображение.")
        return

    # Скачиваем во временный файл
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    await photo_file.download_to_drive(tmp_path)
    logger.info("Скачано изображение: %s (%s)", tmp_path, mime_type)

    try:
        with open(tmp_path, "rb") as f:
            image_bytes = f.read()

        # Формируем содержимое запроса с изображением
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            types.Part.from_text(text=user_prompt),
        ]

        # Добавляем историю диалога (только текстовую часть для контекста)
        history = get_history(user_id)

        response = client.models.generate_content(
            model=MODEL,
            contents=history + [{"role": "user", "parts": [
                {"inline_data": {"mime_type": mime_type, "data": __import__("base64").b64encode(image_bytes).decode()}},
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
        await update.message.reply_text(
            "⚠️ Не удалось обработать изображение. Попробуй ещё раз."
        )
        return
    finally:
        # Удаляем временный файл
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Сохраняем в историю (только текст, без байтов)
    add_message(user_id, "user", [{"text": f"[Изображение] {user_prompt}"}])
    add_message(user_id, "model", [{"text": reply_text}])

    await send_reply(update, reply_text)

# ---------------------------------------------------------------------------
# Текстовые сообщения
# ---------------------------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text
    mode = user_mode.get(user_id, "default")

    relevant_page_nums: list[int] = []

    if mode == "obzr":
        pages = load_book_pages()
        if not pages:
            await update.message.reply_text(
                "⏳ Учебник ещё обрабатывается. Попробуй через несколько минут."
            )
            return
        relevant_page_nums = search_relevant_pages(user_text, pages, top_n=4)
        logger.info("Найдены страницы для '%s': %s", user_text[:50], relevant_page_nums)

        if relevant_page_nums:
            relevant_texts = {n: pages[n] for n in relevant_page_nums if n in pages}
            system_prompt = build_obzr_prompt(relevant_texts)
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

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=8192,
            ),
        )
        reply_text = response.text

    except Exception as exc:
        logger.error("Ошибка Gemini API: %s", exc)
        get_history(user_id).pop()
        await update.message.reply_text(
            "⚠️ Не удалось получить ответ от ИИ. Попробуй ещё раз."
        )
        return

    add_message(user_id, "model", [{"text": reply_text}])
    await send_reply(update, reply_text)

    if mode == "obzr" and relevant_page_nums:
        for page_num in relevant_page_nums[:1]:
            img_path = get_page_image(page_num)
            if img_path:
                try:
                    await context.bot.send_chat_action(
                        chat_id=update.effective_chat.id, action="upload_photo"
                    )
                    with open(img_path, "rb") as img_file:
                        await update.message.reply_photo(
                            photo=img_file,
                            caption=f"📄 Страница {page_num} учебника ОБЖ",
                        )
                except Exception as e:
                    logger.warning("Не удалось отправить страницу %d: %s", page_num, e)

# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Запуск Telegram-бота...")

    _load_user_modes()
    load_book_pages()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("OBZR", cmd_obzr))
    app.add_handler(CommandHandler("obzr", cmd_obzr))
    app.add_handler(CommandHandler("exit", cmd_exit))

    # Обработчик изображений: фото и изображения-документы
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(
        MessageHandler(
            filters.Document.IMAGE,
            handle_photo,
        )
    )

    # Текстовые сообщения
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен. Ожидаю сообщений...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
