FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04


USER root


RUN sudo apt-get update && apt-get -y install



RUN pip install --user pip install --upgrade pip


RUN apt-get -y install git build-essential libglib2.0-0 libsm6 libxext6 ffmpeg libxrender-dev sudo cmake ninja-build

RUN pip install cython
RUN pip install opencv-python-headless
RUN pip install python-multipart


ARG GH_TOKEN

RUN pip install git+https://${GH_TOKEN}@github.com/process-intelligence-research/FlowSheetKnowledgeGraphPI.git

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip install uvicorn gunicorn
RUN pip install fastapi


WORKDIR /app

COPY . .

EXPOSE 8080

ENTRYPOINT ["uvicorn", "api:app", "--proxy-headers",  "--host=0.0.0.0", "--port=80"]