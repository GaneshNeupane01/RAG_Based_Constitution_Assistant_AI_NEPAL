# ← pick a Python base image
FROM python:3.10-slim

# ← set working directory
WORKDIR /app

# → copy everything
COPY . /app

# → install deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# → expose streamlit default port
EXPOSE 8501

# → final command
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
