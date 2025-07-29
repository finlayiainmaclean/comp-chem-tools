FROM mambaorg/micromamba:1.5.7

# # Create bin directory
# RUN mkdir -p /root/bin

# Become root to install system packages
USER root

# Install wget (not included in base)
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV MAMBA_DOCKERFILE_ACTIVATE=1



# Install xtb via micromamba in base environment
RUN micromamba install -y -n base -c conda-forge xtb libgfortran=14.2.0 click

# Download gxtb binary and parameter files into $HOME
RUN wget https://github.com/grimme-lab/g-xtb/raw/refs/heads/main/binary/gxtb -O /opt/conda/bin/gxtb && \
    chmod +x /opt/conda/bin/gxtb && \
    wget https://github.com/grimme-lab/g-xtb/raw/refs/heads/main/parameters/.gxtb -O $HOME/.gxtb && \
    wget https://github.com/grimme-lab/g-xtb/raw/refs/heads/main/parameters/.eeq -O $HOME/.eeq && \
    wget https://github.com/grimme-lab/g-xtb/raw/refs/heads/main/parameters/.basisq -O $HOME/.basisq

# # Optional: make gxtb the default entrypoint
# # ENTRYPOINT ["gxtb"]

# Default to interactive shell
CMD ["/bin/bash"]
