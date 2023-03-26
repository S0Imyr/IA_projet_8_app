FROM python:3.9-slim

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBUG False
ENV SECRET_KEY ''
ENV ALLOWED_HOST ''

# set work directory
WORKDIR /usr/src/app


# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt /usr/src/app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/app

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
