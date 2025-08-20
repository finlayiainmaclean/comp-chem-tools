FROM mambaorg/micromamba:1.5.7

# # Create bin directory
# RUN mkdir -p /root/bin

# Become root to install system packages
USER root



# Install wget (not included in base)
RUN apt-get update && apt-get install -y --no-install-recommends wget nano git build-essential cmake && \
    rm -rf /var/lib/apt/lists/*




# Install xtb via micromamba in base environment
COPY environment.yaml .

RUN micromamba install -y -n base -f environment.yaml && \
    micromamba clean --all --yes


RUN wget https://raw.githubusercontent.com/tblite/tblite/refs/heads/main/python/tblite/ase.py \
    -O /opt/conda/lib/python3.12/site-packages/tblite/ase.py

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV OMP_NUM_THREADS=1



# # Download gxtb binary and parameter files into $HOME
# RUN wget https://github.com/grimme-lab/g-xtb/raw/refs/heads/main/binary/gxtb -O /opt/conda/bin/gxtb && \
#     chmod +x /opt/conda/bin/gxtb && \
#     wget https://github.com/grimme-lab/g-xtb/raw/refs/heads/main/parameters/.gxtb -O $HOME/.gxtb && \
#     wget https://github.com/grimme-lab/g-xtb/raw/refs/heads/main/parameters/.eeq -O $HOME/.eeq && \
#     wget https://github.com/grimme-lab/g-xtb/raw/refs/heads/main/parameters/.basisq -O $HOME/.basisq


WORKDIR /app/cct
COPY pyproject.toml README.rst /app/cct/

COPY cct/__init__.py /app/cct/cct/

RUN pip install -e .
COPY cct /app/cct/cct



RUN echo 'alias jupyter_notebook="jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"' >> ~/.bashrc


# Default to interactive shell
CMD ["/bin/bash"]
