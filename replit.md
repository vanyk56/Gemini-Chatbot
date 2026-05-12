# Gemini Telegram Bot

Telegram-бот-собеседник на базе Google Gemini с памятью контекста, поддержкой изображений и автоматизацией чатов через Telegram Business.

## Run & Operate

- `uv run python bot/main.py` — запустить бота (workflow "Telegram Bot")
- Required secrets: `TELEGRAM_BOT_TOKEN`, `GEMINI_API_KEY`
- Auto-provisioned (Replit): `AI_INTEGRATIONS_GEMINI_BASE_URL`, `AI_INTEGRATIONS_GEMINI_API_KEY`
- Optional: `GEMINI_MODEL_FALLBACK` (default: `gemini-1.5-flash`) — резервная модель при превышении квоты

## Stack

- Python 3.11
- `python-telegram-bot>=22.7` — Telegram Bot API (с поддержкой Business Bot)
- `google-genai>=1.75.0` — Gemini AI SDK

## Where things live

- `bot/main.py` — весь код бота
- `bot/books/obzr.txt` — текст учебника ОБЖ (239 страниц)
- `bot/books/pages/` — страницы учебника в JPG
- `bot/user_modes.json` — режимы пользователей (сохраняется между запусками)
- `bot/user_settings.json` — настройки пользователей

## Architecture decisions

- Gemini клиент поддерживает два режима: Replit proxy (AI_INTEGRATIONS_*) и прямой GEMINI_API_KEY
- Автоматический фоллбэк на `gemini-1.5-flash` при превышении квоты (429 / RESOURCE_EXHAUSTED) — бот не замолкает
- История диалога хранится в памяти, сохраняется покоординатно user/model с временными метками
- Business Bot использует отдельные истории по ключу `biz_{conn_id}_{chat_id}`
- LaTeX автоматически конвертируется в читаемый Unicode-текст перед отправкой
- Системный промпт явно запрещает Gemini использовать LaTeX-нотацию

## Product

- Текстовый ИИ-собеседник с памятью контекста (`/history`, `/settings`, `/reset`)
- Анализ изображений через Gemini Vision (решает задачи, читает текст на фото)
- Режим учебника ОБЖ — строгие ответы только по тексту учебника + фото страницы
- **Автоматизация чатов** — Business Bot отвечает в личных чатах от имени владельца

## User preferences

- Код на Python
- Управление настройками прямо в боте через inline-кнопки (без веб-интерфейса)
- Gemini интеграция через Replit или прямой API ключ

## Gotchas

- Для Business Bot нужен Telegram Premium или Business аккаунт
- После рестарта бота история диалогов теряется (in-memory хранение)
- `filters.UpdateType.BUSINESS_MESSAGE` — правильный фильтр в PTB 22+
- Пакет называется `google-genai`, а не `google-generativeai`
