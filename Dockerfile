FROM ghcr.io/prefix-dev/pixi:0.54.2

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app/cct
COPY pyproject.toml pixi.lock README.rst /app/cct/
COPY cct/__init__.py /app/cct/cct/
# RUN pip install -e .




RUN pixi install && rm -rf ~/.cache/rattler
COPY cct /app/cct/cct
COPY tests /app/cct/tests
COPY benchmarks /app/cct/benchmarks
COPY data /app/cct/data



EXPOSE 8000
RUN echo 'alias jupyter_notebook="jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"' >> ~/.bashrc

CMD [ "pixi", "shell"]
