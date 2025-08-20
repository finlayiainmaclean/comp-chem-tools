import logging
import os
import random
import subprocess
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable

import numpy as np
import ray
import torch
from ase.units import kB
from scipy.special import logsumexp
from tqdm import tqdm

MAX_NUM_CPUS = max(int(cpu_count() * 0.8), 1)

PROJECT_ROOT = Path(__file__).parent.parent


def setup_logger(name, log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()):
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(name)
    return logger


logger = setup_logger(__name__)


def get_best_device():
    """Get the best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def get_cuda_or_cpu_device():
    device = get_best_device()
    if device == "mps":
        device = "cpu"
    return device


device = get_best_device()
print(f"Using device: {device}")


def boltzmann_average_energy(energies, temperature=300.0):
    energies = np.array(energies, dtype=np.float64)
    beta = 1.0 / (kB * temperature)

    # Calculate -beta * E_i
    neg_beta_E = -beta * energies

    # Calculate log partition function
    log_Z = logsumexp(neg_beta_E)

    # Calculate weights
    log_weights = neg_beta_E - log_Z
    weights = np.exp(log_weights)

    return np.sum(energies * weights)


def run_command(cmd, cwd=None, max_attempts=1, retry_delay=0.1, **kwargs):
    """Run a command.

    Args:
        cmd: Command to run
        cwd: Working directory
        max_attempts: Maximum number of attempts for recoverable errors
        retry_delay: Base delay between retries (with jitter)
        **kwargs: Additional arguments passed to subprocess.run
    """

    for attempt in range(max_attempts):
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                capture_output=True,  # Capture both stdout and stderr
                text=True,  # Return strings instead of bytes
                **kwargs,
            )

            # Success - log and return
            logger.debug(f"Command succeeded: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

            # Print stdout if there's content
            if result.stdout:
                logger.debug("Command output:")
                logger.debug(result.stdout.rstrip())

            # Optionally print stderr even on success (some commands output info to stderr)
            if result.stderr:
                logger.warning("Command stderr:")
                logger.warning(result.stderr.rstrip())

            return result

        except subprocess.CalledProcessError as e:
            # Don't retry CalledProcessError - these are actual command failures
            logger.error(f"Command failed with return code {e.returncode}")
            logger.error(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

            if e.stdout:
                logger.error("STDOUT:")
                logger.error(e.stdout)

            if e.stderr:
                logger.error("STDERR:")
                logger.error(e.stderr)

            raise  # Re-raise the exception

        except (OSError, Exception) as e:
            # Retry on OSError (like "Text file busy")
            if attempt < max_attempts - 1:
                if e.errno == 26:  # Text file busy
                    logger.warning(f"Text file busy error, retrying ({attempt + 1}/{max_attempts})")
                else:
                    logger.warning(f"OS error ({e.errno}: {e.strerror}), retrying ({attempt + 1}/{max_attempts})")

                # Add jittered delay to avoid thundering herd
                delay = retry_delay * (2**attempt) + random.uniform(0, 0.1)
                time.sleep(delay)
                continue
            else:
                # Final attempt failed
                logger.error(f"Command failed after {max_attempts} attempts with OS error: {e}")
                logger.error(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
                raise

    # Should never reach here, but just in case
    raise RuntimeError(f"Command failed after {max_attempts} attempts")


def batch_run(
    func: Callable,
    /,
    inputs: list[Any],
    ray_kwargs: Any | None = None,
    run_local: bool = False,
    task_timeout: int = 30,
):
    """Run a function on multiple inputs, either locally or using Ray.

    Args:
    ----
        func: Function to apply to each input.
        inputs: List of input arguments for each function call.
        ray_kwargs: Keyword arguments for Ray configuration.
        run_local: Whether to run locally instead of using Ray.
        task_timeout: Timeout for Ray tasks in seconds.

    Returns:
    -------
        List of results from function calls.

    """
    if run_local:
        results = [func(*input) for input in tqdm(inputs)]
    else:
        CREATED_RAY_CLUSTER = False
        if not ray.is_initialized():
            CREATED_RAY_CLUSTER = True
            ray.init(num_cpus=int(os.environ.get("NUM_CPUS", MAX_NUM_CPUS)))
        inputs = [(i,) if type(i) is not tuple else i for i in inputs]
        pbar = tqdm(total=len(inputs), unit=" ray jobs", mininterval=60, position=0, leave=True)
        ray_func = ray.remote(func)
        if ray_kwargs:
            ray_func = ray_func.options(**ray_kwargs)
        futures = [ray_func.remote(*i) for i in tqdm(inputs)]
        remaining = futures
        total = len(futures)
        while remaining:
            _, remaining = ray.wait(remaining, timeout=task_timeout)
            done = total - len(remaining)
            # Calculate the increment since last update
            increment = done - pbar.n
            if increment > 0:
                pbar.update(increment)

        results = ray.get(futures)
        pbar.close()  # Don't forget to close the progress bar
        if CREATED_RAY_CLUSTER:
            print("Shutting down local ray cluster")
            ray.shutdown()
    return results
