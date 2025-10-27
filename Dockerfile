FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libomp-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && pip install pip-tools

COPY requirements.in .
COPY requirements-dev.in .

RUN pip-compile requirements.in --output-file=requirements.txt && \
    pip-compile requirements-dev.in --output-file=requirements-dev.txt

RUN pip install -U -r requirements.txt -r requirements-dev.txt

RUN python -m ipykernel install \
      --name balgrist-kernel \
      --display-name "Python (Balgrist Docker)" \
      --sys-prefix

RUN jupyter kernelspec remove python3 -f

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]