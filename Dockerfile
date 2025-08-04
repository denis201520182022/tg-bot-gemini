# 1. Используем официальный образ Python 3.11
FROM python:3.11-slim

# 2. Устанавливаем системные зависимости. FFMPEG все еще нужен для pydub.
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 3. Открываем порт для нашего веб-сервера, чтобы Render мог его видеть
EXPOSE 10000

# 4. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 5. Копируем и устанавливаем наши "легкие" зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Копируем весь остальной код нашего бота
COPY . .

# 7. Указываем команду для запуска бота при старте контейнера
CMD ["python", "main.py"]