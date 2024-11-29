FROM python:3.11-slim

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo 'Asia/Shanghai' > /etc/timezone

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -U funasr modelscope huggingface_hub

EXPOSE 7755

CMD ["sh", "start_api_service.sh"]