import subprocess
from pathlib import Path
from typing import Any, Callable, List, Union

import ray
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent


def docker_run_cmd(
    cmd: Union[str, List[str]],
    image: str,
    mount_dir: Union[str, Path],
    workdir: str = "/data",
    auto_remove: bool = True,
    capture_output: bool = True,
    stream_output: bool = True,
    check: bool = True,
):
    """Run a command inside a Docker container using the Docker SDK.

    Args:
    ----
        cmd: Command to run inside the container.
        image: Docker image to use.
        mount_dir: Host directory to mount.
        workdir: Working directory inside container.
        auto_remove: Automatically remove container after run.
        capture_output: Return stdout/stderr as strings.
        stream_output: Print output live.
        check: Raise exception on non-zero exit.

    Returns:
    -------
        dict with keys: 'exit_code', 'stdout', 'stderr'

    """
    import docker

    client = docker.from_env()

    mount_path = Path(mount_dir).resolve()
    volumes = {str(mount_path): {"bind": workdir, "mode": "rw"}}

    if isinstance(cmd, list):
        pass  # leave as-is
    elif isinstance(cmd, str):
        cmd = ["/bin/bash", "-c", cmd]  # run as bash script
    else:
        raise ValueError("cmd must be a str or list of str")

    container = client.containers.run(
        image=image,
        command=cmd,
        volumes=volumes,
        working_dir=workdir,
        detach=True,
        auto_remove=auto_remove,
        stderr=True,
        stdout=True,
    )

    stdout, stderr = "", ""

    if capture_output:
        if stream_output:
            for line in container.logs(stream=True):
                print(line.decode(), end="")
        else:
            logs = container.logs(stdout=True, stderr=True)
            stdout = logs.decode()

    exit_code = container.wait()["StatusCode"]

    if check and exit_code != 0:
        raise RuntimeError(
            f"Docker command failed with code {exit_code}: {stderr or stdout}"
        )

    return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}


def run_command(cmd, cwd=None, **kwargs):
    """Run a command and only print stdout/stderr if there's an error."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,  # Capture both stdout and stderr
            text=True,  # Return strings instead of bytes
            **kwargs,
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

        if e.stdout:
            print("STDOUT:")
            print(e.stdout)

        if e.stderr:
            print("STDERR:")
            print(e.stderr)

        raise  # Re-raise the exception


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
        print("Creating locally sequentially")
        results = [func(*input) for input in inputs]
    else:
        CREATED_RAY_CLUSTER = False
        if not ray.is_initialized():
            print("Creating local ray cluster")
            CREATED_RAY_CLUSTER = True
            ray.init()
        inputs = [(i,) if type(i) is not tuple else i for i in inputs]
        pbar = tqdm(
            total=len(inputs), unit=" ray jobs", mininterval=60, position=0, leave=True
        )
        ray_func = ray.remote(func)
        if ray_kwargs:
            ray_func = ray_func.options(**ray_kwargs)
        futures = [ray_func.remote(*i) for i in inputs]
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
