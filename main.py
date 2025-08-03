import asyncio
import logging
import os
import re

from aiogram import Bot, Dispatcher, types, Router
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (InlineKeyboardButton, InlineKeyboardMarkup,
                           KeyboardButton, ReplyKeyboardMarkup,
                           ReplyKeyboardRemove)
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------ Конфигурация ------------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
- Всегда экранируй следующие специальные символы, добавляя перед ними обратный слэш (\\): . ! - = + ( ) { } [ ] | # > _ * ~
- **Пример ПРАВИЛЬНОГО экранирования:** Вместо "Файл main.py" нужно писать "Файл main\\.py". Вместо "Идея-стартап!" нужно писать "Идея\\-стартап\\!". Вместо "(GPT-3)" нужно писать "\\(GPT\\-3\\)".
- НЕ НУЖНО экранировать символ обратной кавычки (`). Используй его только для форматирования кода.
"""

# ------------------ Клавиатуры ------------------
button_new_request = KeyboardButton(text="✅ Новый запрос")
keyboard = ReplyKeyboardMarkup(keyboard=[[button_new_request]], resize_keyboard=True, input_field_placeholder="Задайте вопрос...")

# --- ИЗМЕНЕНИЯ ЗДЕСЬ ---
button_idea = InlineKeyboardButton(text="💡 Идея для стартапа", callback_data="idea")
button_poem = InlineKeyboardButton(text="✍️ Напиши стих", callback_data="poem")
button_story = InlineKeyboardButton(text="📝 Написать рассказ", callback_data="story")
button_travel = InlineKeyboardButton(text="✈️ Спланировать путешествие", callback_data="travel")
button_recipe = InlineKeyboardButton(text="🍳 Рецепт по ингредиентам", callback_data="recipe")

# Размещаем кнопки в более удобной сетке 2x2 + 1
inline_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [button_idea, button_poem],
    [button_story, button_travel],
    [button_recipe]
])

# ------------------ Клиент Gemini ------------------
class GeminiClient:
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            logging.critical(f"Ошибка конфигурации Gemini: {e}")
            self.model = None

    async def generate_text(self, prompt: str) -> str:
        if not self.model:
            return "Ошибка: Модель Gemini не была инициализирована. Проверьте API ключ."
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            return response.text
        except Exception as e:
            logging.error(f"Ошибка при генерации текста Gemini: {e}")
            return "Извините, произошла ошибка при генерации ответа. Попробуйте позже."

# ------------------ Инициализация ------------------
if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Не найдены переменные окружения TELEGRAM_TOKEN или GEMINI_API_KEY")

gemini_client = GeminiClient(GEMINI_API_KEY)
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
    # --- ИЗМЕНЕНИЯ ЗДЕСЬ ---
    waiting_for_story_prompt = State()
    waiting_for_travel_details = State()
    waiting_for_ingredients = State()


# ------------------ Вспомогательные функции ------------------

def sanitize_markdown_v2(text: str) -> str:
    """
    Принудительно экранирует некоторые критичные символы для MarkdownV2, 
    которые ИИ часто забывает экранировать.
    """
    escape_chars = r'._*[]()~>#+-=|{}!' 
    for char in escape_chars:
        pattern = r'(?<!\\)' + re.escape(char)
        replacement = r'\\' + char
        text = re.sub(pattern, replacement, text)
    return text

async def prepare_prompt_with_system_message(prompt: str, state: FSMContext) -> str:
    data = await state.get_data()
    user_system_prompt = data.get("system_prompt")
    full_system_prompt_parts = [DEFAULT_SYSTEM_PROMPT]
    if user_system_prompt:
        full_system_prompt_parts.append(f"Дополнительная инструкция от пользователя: {user_system_prompt}")
    final_system_part = "\n\n---\n\n".join(full_system_prompt_parts)
    return f"{final_system_part}\n\n===\n\nЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{prompt}"

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
        logging.warning("Ошибка форматирования MarkdownV2 даже после очистки! Отправка простого текста.")
        logging.info(f"--- ПРОБЛЕМНЫЙ (ОЧИЩЕННЫЙ) ТЕКСТ ---\n{sanitized_text}\n--- КОНЕЦ ТЕКСТА ---")
        for i in range(0, len(text), 4096):
            chunk = text[i:i + 4096]
            await message.answer(chunk, reply_markup=keyboard)
    except Exception as e:
        logging.exception(f"Неизвестная ошибка при отправке сообщения: {e}")
        await message.answer("Произошла критическая ошибка при отправке ответа.", reply_markup=keyboard)

# ------------------ ОБРАБОТЧИКИ ------------------

@router.message(Command(commands=["start", "help", "setting"]))
async def handle_commands(message: types.Message, state: FSMContext):
    command = message.text.split()[0] 
    
    if command == "/start":
        await state.clear()
        await message.answer("Привет! Я бот на основе Gemini. Задайте вопрос или выберите опцию:", reply_markup=inline_keyboard)
    elif command == "/help":
        help_text = (
            "*Справка по боту*\n\n"
            "`/start` \\- Перезапустить бота и сбросить контекст\\.\n"
            "`/help` \\- Показать это сообщение\\.\n"
            "`/setting` \\- Задать *дополнительную* системную инструкцию \\(роль\\) для ИИ\\.\n\n"
            "Кнопка '✅ Новый запрос' также сбрасывает контекст чата\\."
        )
        await message.answer(help_text, parse_mode=ParseMode.MARKDOWN_V2)
    elif command == "/setting":
        await state.set_state(Form.waiting_for_system_prompt)
        await message.answer("Введите дополнительную системную инструкцию (например, 'Ты — шеф-повар').", reply_markup=ReplyKeyboardRemove())

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
    await state.set_state(None)
    user_prompt = f"Придумай и подробно опиши идею для стартапа в сфере: {message.text}"
    full_prompt = await prepare_prompt_with_system_message(user_prompt, state)
    gpt_response = await gemini_client.generate_text(full_prompt)
    await send_formatted_answer(message, gpt_response)

@router.message(StateFilter(Form.waiting_for_poem_topic))
async def process_poem_topic(message: types.Message, state: FSMContext):
    await state.set_state(None)
    user_prompt = f"Напиши красивый стих на тему: {message.text}"
    full_prompt = await prepare_prompt_with_system_message(user_prompt, state)
    gpt_response = await gemini_client.generate_text(full_prompt)
    await send_formatted_answer(message, gpt_response)

# --- ИЗМЕНЕНИЯ ЗДЕСЬ: НОВЫЕ ОБРАБОТЧИКИ СОСТОЯНИЙ ---
@router.message(StateFilter(Form.waiting_for_story_prompt))
async def process_story_prompt(message: types.Message, state: FSMContext):
    await state.set_state(None)
    user_prompt = f"Напиши интересный короткий рассказ на тему: {message.text}"
    full_prompt = await prepare_prompt_with_system_message(user_prompt, state)
    gpt_response = await gemini_client.generate_text(full_prompt)
    await send_formatted_answer(message, gpt_response)

@router.message(StateFilter(Form.waiting_for_travel_details))
async def process_travel_details(message: types.Message, state: FSMContext):
    await state.set_state(None)
    user_prompt = f"Составь подробный и интересный план путешествия. Детали от пользователя: {message.text}. Предложи места для посещения, тайминг и возможные активности."
    full_prompt = await prepare_prompt_with_system_message(user_prompt, state)
    gpt_response = await gemini_client.generate_text(full_prompt)
    await send_formatted_answer(message, gpt_response)

@router.message(StateFilter(Form.waiting_for_ingredients))
async def process_ingredients(message: types.Message, state: FSMContext):
    await state.set_state(None)
    user_prompt = f"Придумай подробный рецепт, используя следующие ингредиенты: {message.text}. Укажи название блюда, полный список нужных ингредиентов (включая те, что я назвал), пошаговую инструкцию и примерное время приготовления."
    full_prompt = await prepare_prompt_with_system_message(user_prompt, state)
    gpt_response = await gemini_client.generate_text(full_prompt)
    await send_formatted_answer(message, gpt_response)
# --- КОНЕЦ НОВЫХ ОБРАБОТЧИКОВ ---

@router.callback_query(lambda c: c.data in ["idea", "poem", "story", "travel", "recipe"])
async def process_callbacks(callback_query: types.CallbackQuery, state: FSMContext):
    await bot.answer_callback_query(callback_query.id)
    if callback_query.data == "idea":
        await bot.send_message(callback_query.from_user.id, "В какой сфере вы хотите идею для стартапа?", reply_markup=ReplyKeyboardRemove())
        await state.set_state(Form.waiting_for_startup_area)
    elif callback_query.data == "poem":
        await bot.send_message(callback_query.from_user.id, "На какую тему написать стих?", reply_markup=ReplyKeyboardRemove())
        await state.set_state(Form.waiting_for_poem_topic)
    # --- ИЗМЕНЕНИЯ ЗДЕСЬ ---
    elif callback_query.data == "story":
        await bot.send_message(callback_query.from_user.id, "О чем написать рассказ? Задайте тему или персонажей.", reply_markup=ReplyKeyboardRemove())
        await state.set_state(Form.waiting_for_story_prompt)
    elif callback_query.data == "travel":
        await bot.send_message(callback_query.from_user.id, "Куда и на сколько дней вы хотите поехать?", reply_markup=ReplyKeyboardRemove())
        await state.set_state(Form.waiting_for_travel_details)
    elif callback_query.data == "recipe":
        await bot.send_message(callback_query.from_user.id, "Какие ингредиенты у вас есть? (перечислите через запятую)", reply_markup=ReplyKeyboardRemove())
        await state.set_state(Form.waiting_for_ingredients)

@router.message()
async def chat_with_gpt(message: types.Message, state: FSMContext):
    data = await state.get_data()
    history = data.get('chat_history', [])
    history.append(f"Пользователь: {message.text}")
    chat_prompt = "\n\n".join(history[-7:])
    full_prompt = await prepare_prompt_with_system_message(chat_prompt, state)
    gpt_response = await gemini_client.generate_text(full_prompt)
    history.append(f"Бот: {gpt_response}")
    await state.update_data(chat_history=history[-10:])
    await send_formatted_answer(message, gpt_response)

# ------------------ Запуск бота ------------------
async def main():
    commands = [
        types.BotCommand(command="/start", description="Начать работу"),
        types.BotCommand(command="/help", description="Справка"),
        types.BotCommand(command="/setting", description="Задать доп. инструкцию")
    ]
    await bot.set_my_commands(commands)
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Бот остановлен.")