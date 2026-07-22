FROM kathara/base
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install  --break-system-packages --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu