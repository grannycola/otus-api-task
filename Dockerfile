FROM python:3.8-slim

# Установите рабочий каталог в контейнере
WORKDIR /app

# Копируйте файлы зависимостей и устанавливайте их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируйте исходный код вашего приложения в контейнер
COPY . .

# Команда для запуска приложения
CMD ["uvicorn", "webapp_classifier:app", "--host", "0.0.0.0", "--port", "80"]