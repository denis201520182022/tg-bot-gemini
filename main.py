import asyncio
import logging
import os

import aiogram
from aiogram import Bot, Dispatcher, types, Router
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardRemove, ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


logging.basicConfig(level=logging.INFO)


button_new_request = KeyboardButton(text="✅ Новый запрос")

keyboard = ReplyKeyboardMarkup(keyboard=[
    [button_new_request]
], resize_keyboard=True, input_field_placeholder="Задайте вопрос...")

# ------------------ Gemini Client Class ------------------
import google.generativeai as genai

class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.0-flash-exp")

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                return response.text
            else:
                logging.error(f"Ошибка при генерации текста: {response}")
                return "Произошла ошибка при генерации ответа."
        except Exception as e:
            logging.error(f"Ошибка при запросе к Gemini: {e}")
            return "Произошла ошибка при обработке запроса. Попробуйте еще раз позже."


gemini_client = GeminiClient(GEMINI_API_KEY)

storage = MemoryStorage()

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(storage=storage)

router = Router()
dp.include_router(router)



@router.message(Command(commands=["start"]))
async def start_command(message: types.Message, state: FSMContext):
    """Обработчик команды /start."""
    await state.set_state(None)  # Сброс контекста
    await message.answer("Привет! Я бот, работающий на Gemini. Задайте свой вопрос.", reply_markup=keyboard)


@router.message(Command(commands=["help"]))
async def help_command(message: types.Message):
    """Обработчик команды /help."""
    await message.answer("Я бот, который использует Gemini для ответов на ваши вопросы.\n"
                        "Просто напишите свой вопрос, и я постараюсь ответить.\n"
                        "Используйте кнопку 'Новый запрос' для сброса контекста.", reply_markup=keyboard)


@router.message(lambda message: message.text == "✅ Новый запрос")
async def new_request(message: types.Message, state: FSMContext):
    """Обработчик кнопки "Новый запрос"."""
    await state.set_state(None)
    await message.answer("Контекст сброшен. Задайте свой вопрос.", reply_markup=keyboard)

@router.message()
async def chat_with_gpt(message: types.Message, state: FSMContext):
    """Обработчик всех текстовых сообщений."""
    user_id = message.from_user.id

    # Получаем историю диалога из состояния
    try:
        data = await state.get_data()
        if 'history' not in data:
            data['history'] = []
        history = data['history']
    except Exception as e:
        logging.exception("Failed to get history from FSM:")
        history=[]

    
    history.append({"role": "user", "content": message.text})

    try:
        # Формируем строку запроса для Gemini
        prompt = "\n".join([m['content'] for m in history])

        
        gpt_response = gemini_client.generate_text(prompt)  # Используем класс GeminiClient

        
        history.append({"role": "assistant", "content": gpt_response})

        await message.answer(gpt_response, reply_markup=keyboard)

    except Exception as e:
        logging.exception(f"Произошла ошибка при запросе к Gemini: {e}")
        await message.answer("Произошла ошибка при обработке запроса. Попробуйте еще раз позже.", reply_markup=keyboard)

    
    try:
        await state.update_data(history=history)
    except Exception as e:
        logging.exception("Failed to update FSM data:")


async def main():
    try:
        
        commands = [
            types.BotCommand(command="/start", description="Начать работу с ботом"),
            types.BotCommand(command="/help", description="Получить справку о боте")
        ]
        await bot.set_my_commands(commands)

        await dp.start_polling(bot, reset_webhook=True)
    except asyncio.CancelledError:
        logging.info("Бот остановлен из-за отмены задачи")
    except KeyboardInterrupt:
        logging.info("Бот остановлен нажатием Ctrl+C")
    finally:
        await bot.session.close()

if __name__ == '__main__':
    asyncio.run(main())