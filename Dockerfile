# Базовый образ Python
FROM python:3.11-slim

# Установка системных зависимостей (если потребуется сборка)
RUN apt-get update && apt-get install -y build-essential curl

# Создаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Переменные окружения (можно убрать и использовать .env файл)
ENV TELEGRAM_TOKEN=8321906655:AAH29bFDJTJuTT5RAcyhT8KtIe3y8CMVAdA
ENV GEMINI_API_KEY=AIzaSyBaOMBX2anhPUV9tUtwC2taP1KJ3QsFmiM

# Запуск бота
CMD ["python", "main.py"]
