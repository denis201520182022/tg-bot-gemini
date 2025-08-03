import asyncio
import logging
import os
import re
from threading import Thread
from flask import Flask

from aiogram import Bot, Dispatcher, types, Router, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (InlineKeyboardButton, InlineKeyboardMarkup,
                           KeyboardButton, ReplyKeyboardMarkup,
                           ReplyKeyboardRemove, FSInputFile)
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
import torch
import whisper

# ------------------ Конфигурация ------------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not os.path.exists("temp"):
    os.makedirs("temp")

logging.basicConfig(level=logging.INFO)

# ------------------ ДЕФОЛТНАЯ СИСТЕМНАЯ ИНСТРУКЦИЯ ------------------
DEFAULT_SYSTEM_PROMPT = """
Ты — полезный ассистент в Telegram. Твоя задача — отвечать на запросы пользователя.
АБСОЛЮТНО ВСЕГДА, без исключений, форматируй свой ответ, используя синтаксис MarkdownV2 для Telegram.

**Правила форматирования:**
- Жирный: **текст**
- Курсив: *текст*
- Подчеркнутый: __текст__
- Зачеркнутый: ~текст~
- Моноширинный код (инлайн): `текст`
- Блок с кодом: ```python\nкод\n```
- Ссылки: [текст](URL)

**ЗАПРЕЩЕНО:**
- Категорически запрещено использовать заголовки с помощью символов #.

**ОЧЕНЬ ВАЖНО (ЭКРАНИРОВАНИЕ):**
- Всегда экранируй следующие специальные символы, добавляя перед ними обратный слэш (\\\\): `._*[]()~>#+-=|{}!`
"""

# ------------------ Клавиатуры ------------------
button_new_request = KeyboardButton(text="✅ Новый запрос")
keyboard = ReplyKeyboardMarkup(keyboard=[[button_new_request]], resize_keyboard=True, input_field_placeholder="Задайте вопрос...")
button_idea = InlineKeyboardButton(text="💡 Идея для стартапа", callback_data="idea")
button_poem = InlineKeyboardButton(text="✍️ Напиши стих", callback_data="poem")
button_story = InlineKeyboardButton(text="📝 Написать рассказ", callback_data="story")
button_travel = InlineKeyboardButton(text="✈️ Спланировать путешествие", callback_data="travel")
button_recipe = InlineKeyboardButton(text="🍳 Рецепт по ингредиентам", callback_data="recipe")
inline_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [button_idea, button_poem],
    [button_story, button_travel],
    [button_recipe]
])

# ------------------ Клиенты AI ------------------
class GeminiClient:
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            logging.critical(f"Ошибка конфигурации Gemini: {e}")
            self.model = None

    async def generate_text(self, prompt: str) -> str:
        if not self.model: return "Ошибка: Модель Gemini не была инициализирована."
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            return response.text
        except Exception as e:
            logging.error(f"Ошибка при генерации текста Gemini: {e}")
            return "Извините, произошла ошибка при генерации ответа."

class LocalSpeechClient:
    def __init__(self, model_name="base"):
        self.model = None
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Загрузка локальной модели Whisper '{model_name}' на устройстве '{device}'...")
            self.model = whisper.load_model(model_name, device=device)
            logging.info("Модель Whisper успешно загружена.")
        except Exception as e:
            logging.critical(f"Не удалось загрузить локальную модель Whisper: {e}\nУбедитесь, что у вас установлены 'torch' и 'openai-whisper'.")

    async def audio_to_text(self, audio_filepath: str) -> str | None:
        if not self.model: return None
        try:
            use_fp16 = self.model.device.type == 'cuda'
            result = await asyncio.to_thread(self.model.transcribe, audio_filepath, fp16=use_fp16, language='ru')
            logging.info(f"Распознанный текст: {result['text']}")
            return result['text']
        except Exception as e:
            logging.error(f"Ошибка при локальной транскрибации аудио: {e}")
            return None

class TTSClient:
    async def text_to_audio(self, text: str, output_filepath: str) -> bool:
        try:
            clean_text = re.sub(r'[*_`~[\]()\\#+-.!{}]', '', text).replace("```", " ").replace("`", " ")
            if not clean_text.strip(): return False
            def sync_tts():
                gTTS(clean_text, lang='ru').save(output_filepath)
                return True
            return await asyncio.to_thread(sync_tts)
        except Exception as e:
            logging.error(f"Ошибка при синтезе речи (gTTS): {e}")
            return False

# ------------------ Инициализация ------------------
if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Не найдены переменные окружения TELEGRAM_TOKEN или GEMINI_API_KEY")

gemini_client = GeminiClient(GEMINI_API_KEY)
local_speech_client = LocalSpeechClient(model_name="base")
tts_client = TTSClient()
storage = MemoryStorage()
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(storage=storage)
router = Router()
dp.include_router(router)

# ------------------ Состояния FSM ------------------
class Form(StatesGroup):
    waiting_for_startup_area = State()
    waiting_for_poem_topic = State()
    waiting_for_system_prompt = State()
    waiting_for_story_prompt = State()
    waiting_for_travel_details = State()
    waiting_for_ingredients = State()

# ------------------ Вспомогательные функции ------------------
def sanitize_markdown_v2(text: str) -> str:
    escape_chars = r'._*[]()~>#+-=|{}!'
    for char in escape_chars:
        pattern = r'(?<!\\)' + re.escape(char)
        replacement = r'\\' + char
        text = re.sub(pattern, replacement, text)
    return text

async def build_gemini_prompt(history: list, new_prompt: str, state: FSMContext) -> str:
    data = await state.get_data()
    user_system_prompt = data.get("system_prompt")
    
    full_system_prompt_parts = [DEFAULT_SYSTEM_PROMPT]
    if user_system_prompt:
        full_system_prompt_parts.append(f"Дополнительная инструкция от пользователя: {user_system_prompt}")
    
    system_part = "\n\n---\n\n".join(full_system_prompt_parts)

    history_part = ""
    if history:
        formatted_history = "\n".join(history)
        history_part = f"""
---
**КОНТЕКСТ ПРЕДЫДУЩЕГО ДИАЛОГА (для справки):**
{formatted_history}
---
"""
    new_prompt_part = f"""
**АКТУАЛЬНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ (ответь только на него):**
{new_prompt}
"""
    return f"{system_part}{history_part}{new_prompt_part}"

async def send_formatted_answer(message: types.Message, text: str):
    if not text:
        await message.answer("К сожалению, я не смог сгенерировать ответ.", reply_markup=keyboard)
        return
    sanitized_text = sanitize_markdown_v2(text)
    try:
        for i in range(0, len(sanitized_text), 4096):
            chunk = sanitized_text[i:i + 4096]
            await message.answer(chunk, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard)
    except TelegramBadRequest:
        logging.warning("Ошибка форматирования MarkdownV2. Отправка простого текста.")
        for i in range(0, len(text), 4096):
            chunk = text[i:i + 4096]
            await message.answer(chunk, reply_markup=keyboard)
    except Exception as e:
        logging.error(f"Неизвестная ошибка при отправке сообщения: {e}")
        await message.answer("Произошла критическая ошибка при отправке ответа.", reply_markup=keyboard)

# ------------------ ОБРАБОТЧИКИ ------------------
@router.message(Command(commands=["start", "help", "setting"]))
async def handle_commands(message: types.Message, state: FSMContext):
    command = message.text.split()[0]
    if command == "/start":
        await state.clear()
        await message.answer("Привет! Я бот на основе Gemini. Задайте вопрос текстом, голосом или выберите опцию:", reply_markup=inline_keyboard)
    elif command == "/help":
        help_text = "*Справка по боту*\n\n`/start` \\- Перезапустить бота и сбросить контекст\\.\n`/help` \\- Показать это сообщение\\.\n`/setting` \\- Задать *дополнительную* системную инструкцию \\(роль\\) для ИИ\\.\n\nВы можете общаться со мной как текстом, так и *голосовыми сообщениями*\\."
        await message.answer(help_text, parse_mode=ParseMode.MARKDOWN_V2)
    elif command == "/setting":
        await state.set_state(Form.waiting_for_system_prompt)
        await message.answer("Введите дополнительную системную инструкцию.", reply_markup=ReplyKeyboardRemove())

@router.message(lambda message: message.text == "✅ Новый запрос")
async def new_request(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("Контекст сброшен. Можете задать новый вопрос.", reply_markup=keyboard)

@router.message(StateFilter(Form.waiting_for_system_prompt))
async def process_system_prompt(message: types.Message, state: FSMContext):
    await state.update_data(system_prompt=message.text)
    await state.set_state(None)
    await message.answer("✅ Дополнительная системная инструкция установлена.", reply_markup=keyboard)

@router.message(StateFilter(Form.waiting_for_startup_area))
async def process_startup_area(message: types.Message, state: FSMContext):
    thinking_message = await message.answer("💡 Генерирую идею для стартапа\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    await state.set_state(None)
    user_prompt = f"Придумай и подробно опиши идею для стартапа в сфере: {message.text}"
    full_prompt = await build_gemini_prompt([], user_prompt, state)
    gpt_response = await gemini_client.generate_text(full_prompt)
    await thinking_message.delete()
    await send_formatted_answer(message, gpt_response)

@router.message(StateFilter(Form.waiting_for_poem_topic))
async def process_poem_topic(message: types.Message, state: FSMContext):
    thinking_message = await message.answer("✍️ Пишу стих\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    await state.set_state(None)
    user_prompt = f"Напиши красивый стих на тему: {message.text}"
    full_prompt = await build_gemini_prompt([], user_prompt, state)
    gpt_response = await gemini_client.generate_text(full_prompt)
    await thinking_message.delete()
    await send_formatted_answer(message, gpt_response)

@router.message(StateFilter(Form.waiting_for_story_prompt))
async def process_story_prompt(message: types.Message, state: FSMContext):
    thinking_message = await message.answer("📝 Сочиняю рассказ\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    await state.set_state(None)
    user_prompt = f"Напиши интересный короткий рассказ на тему: {message.text}"
    full_prompt = await build_gemini_prompt([], user_prompt, state)
    gpt_response = await gemini_client.generate_text(full_prompt)
    await thinking_message.delete()
    await send_formatted_answer(message, gpt_response)

@router.message(StateFilter(Form.waiting_for_travel_details))
async def process_travel_details(message: types.Message, state: FSMContext):
    thinking_message = await message.answer("✈️ Планирую путешествие\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    await state.set_state(None)
    user_prompt = f"Составь подробный и интересный план путешествия. Детали от пользователя: {message.text}."
    full_prompt = await build_gemini_prompt([], user_prompt, state)
    gpt_response = await gemini_client.generate_text(full_prompt)
    await thinking_message.delete()
    await send_formatted_answer(message, gpt_response)

@router.message(StateFilter(Form.waiting_for_ingredients))
async def process_ingredients(message: types.Message, state: FSMContext):
    thinking_message = await message.answer("🍳 Ищу рецепт\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    await state.set_state(None)
    user_prompt = f"Придумай подробный рецепт, используя следующие ингредиенты: {message.text}."
    full_prompt = await build_gemini_prompt([], user_prompt, state)
    gpt_response = await gemini_client.generate_text(full_prompt)
    await thinking_message.delete()
    await send_formatted_answer(message, gpt_response)

@router.callback_query(lambda c: c.data in ["idea", "poem", "story", "travel", "recipe"])
async def process_callbacks(callback_query: types.CallbackQuery, state: FSMContext):
    await bot.answer_callback_query(callback_query.id)
    actions = {
        "idea": ("В какой сфере вы хотите идею для стартапа?", Form.waiting_for_startup_area),
        "poem": ("На какую тему написать стих?", Form.waiting_for_poem_topic),
        "story": ("О чем написать рассказ?", Form.waiting_for_story_prompt),
        "travel": ("Куда и на сколько дней вы хотите поехать?", Form.waiting_for_travel_details),
        "recipe": ("Какие ингредиенты у вас есть? (перечислите через запятую)", Form.waiting_for_ingredients)
    }
    text, new_state = actions[callback_query.data]
    await bot.send_message(callback_query.from_user.id, text, reply_markup=ReplyKeyboardRemove())
    await state.set_state(new_state)

@router.message(F.voice)
async def handle_voice(message: types.Message, state: FSMContext):
    processing_message = await message.answer("️️🎙️ Обрабатываю голосовое сообщение\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    voice_file_id = message.voice.file_id
    oga_filepath = os.path.join("temp", f"{voice_file_id}.oga")
    mp3_filepath = os.path.join("temp", f"{voice_file_id}.mp3")
    response_audio_path = os.path.join("temp", f"response_{voice_file_id}.mp3")

    try:
        file_info = await bot.get_file(voice_file_id)
        await bot.download_file(file_info.file_path, destination=oga_filepath)
        audio = await asyncio.to_thread(AudioSegment.from_file, oga_filepath, format="ogg")
        await asyncio.to_thread(audio.export, mp3_filepath, format="mp3")

        user_text = await local_speech_client.audio_to_text(mp3_filepath)
        if not user_text:
            await processing_message.edit_text("Не удалось распознать речь в сообщении\\.", parse_mode=ParseMode.MARKDOWN_V2)
            return

        sanitized_user_text = sanitize_markdown_v2(user_text)
        await processing_message.edit_text(f"Вы сказали: *«{sanitized_user_text}»*\n\n🧠 Думаю над ответом\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)

        data = await state.get_data()
        history = data.get('chat_history', [])

        full_prompt = await build_gemini_prompt(history, user_text, state)
        raw_response = await gemini_client.generate_text(full_prompt)
        clean_response = raw_response.strip()
        if clean_response.lower().startswith("бот:"):
            clean_response = clean_response[4:].lstrip()

        history.append(f"Пользователь: {user_text}")
        history.append(f"Бот: {clean_response}")
        await state.update_data(chat_history=history[-10:])

        await processing_message.delete()
        await send_formatted_answer(message, clean_response)

        tts_success = await tts_client.text_to_audio(clean_response, response_audio_path)
        if tts_success and os.path.exists(response_audio_path):
            await message.answer_voice(FSInputFile(response_audio_path))

    except Exception as e:
        logging.error(f"Полная ошибка в handle_voice: {e}")
        try: await processing_message.edit_text("Произошла непредвиденная ошибка при обработке\\.", parse_mode=ParseMode.MARKDOWN_V2)
        except TelegramBadRequest: pass
    finally:
        for f in [oga_filepath, mp3_filepath, response_audio_path]:
            if os.path.exists(f): os.remove(f)

@router.message(F.text)
async def chat_with_gpt(message: types.Message, state: FSMContext):
    thinking_message = await message.answer("🧠 Думаю над вашим запросом\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
    try:
        data = await state.get_data()
        history = data.get('chat_history', [])
        
        full_prompt = await build_gemini_prompt(history, message.text, state)
        
        raw_response = await gemini_client.generate_text(full_prompt)
        clean_response = raw_response.strip()
        if clean_response.lower().startswith("бот:"):
            clean_response = clean_response[4:].lstrip()
        
        history.append(f"Пользователь: {message.text}")
        history.append(f"Бот: {clean_response}")
        await state.update_data(chat_history=history[-10:])

        await thinking_message.delete()
        await send_formatted_answer(message, clean_response)

    except Exception as e:
        logging.error(f"Полная ошибка в chat_with_gpt: {e}")
        try: await thinking_message.edit_text("Произошла непредвиденная ошибка при обработке\\.", parse_mode=ParseMode.MARKDOWN_V2)
        except TelegramBadRequest: pass

# ------------------ Запуск бота и веб-сервера ------------------
app = Flask(__name__)

@app.route('/')
def index():
    return "I'm alive!"

def run_web_server():
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

async def start_bot():
    commands = [
        types.BotCommand(command="/start", description="Начать работу"),
        types.BotCommand(command="/help", description="Справка"),
        types.BotCommand(command="/setting", description="Задать доп. инструкцию")
    ]
    await bot.set_my_commands(commands)
    await dp.start_polling(bot)

if __name__ == '__main__':
    web_thread = Thread(target=run_web_server)
    web_thread.start()
    
    try:
        logging.info("Starting bot polling...")
        asyncio.run(start_bot())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped.")