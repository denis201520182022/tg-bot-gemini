# 1. Используем официальный образ Python 3.11
FROM python:3.11-slim

# 2. Устанавливаем системные зависимости, в первую очередь FFMPEG
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 3. Открываем порт для нашего веб-сервера, чтобы Render мог его видеть
EXPOSE 10000

# 4. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 5. Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Предварительно скачиваем модель Whisper, чтобы бот стартовал быстро
# Этот шаг может занять несколько минут во время сборки!
RUN python -c "import whisper; whisper.load_model('base')"

# 7. Копируем весь остальной код нашего бота в контейнер
COPY . .

# 8. Указываем команду для запуска бота при старте контейнера
CMD ["python", "main.py"]