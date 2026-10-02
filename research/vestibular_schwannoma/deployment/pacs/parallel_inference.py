"""CPU-parallel model-ensemble inference for the PACS deployment."""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import queue
import tempfile
import time
import traceback
from pathlib import Path
from threading import BrokenBarrierError

import numpy as np
import torch
import torchio as tio
from fastMONAI.vision_all import load_safetensors_model
from fastMONAI.vision_patch import PatchInferenceEngine


THREADS_PER_MODEL = 3
WORKER_STARTUP_TIMEOUT_SECONDS = 600
WORKER_FORWARD_TIMEOUT_SECONDS = 900
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 30
NO_FLIP_COMMAND = -1
STOP_COMMAND = -2

# Batch tensors have shape [B, C, D, H, W], so spatial axes are 2, 3, and 4.
TTA_FLIP_AXES = (
    (),
    (4,),
    (3,),
    (2,),
    (3, 4),
    (2, 4),
    (2, 3),
    (2, 3, 4),
)


def assign_worker_cpus(
    available_cpus, worker_count: int, threads_per_model: int = THREADS_PER_MODEL
) -> list[tuple[int, ...]]:
    """Return disjoint fixed-size CPU sets for ensemble workers."""
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if threads_per_model < 1:
        raise ValueError("threads_per_model must be positive")
    available = tuple(sorted(set(available_cpus)))
    required = worker_count * threads_per_model
    if len(available) < required:
        raise RuntimeError(
            f"parallel {worker_count}-model ensemble requires at least {required} "
            f"available CPU threads ({threads_per_model} per model), found "
            f"{len(available)}"
        )
    return [
        available[index : index + threads_per_model]
        for index in range(0, required, threads_per_model)
    ]


def _logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=1)


def _ensemble_worker(
    member_id,
    model_path,
    cpu_set,
    input_path,
    input_shape,
    output_path,
    output_shape,
    output_offset,
    command,
    barrier,
    ready_queue,
    error_queue,
):
    input_array = None
    output_array = None
    try:
        os.sched_setaffinity(0, set(cpu_set))
        thread_count = len(cpu_set)
        os.environ["OMP_NUM_THREADS"] = str(thread_count)
        os.environ["MKL_NUM_THREADS"] = str(thread_count)
        torch.set_num_threads(thread_count)
        torch.set_num_interop_threads(1)

        input_array = np.memmap(
            input_path, dtype=np.float32, mode="r+", shape=tuple(input_shape)
        )
        output_array = np.memmap(
            output_path,
            dtype=np.float32,
            mode="r+",
            offset=output_offset,
            shape=tuple(output_shape),
        )
        input_tensor = torch.from_numpy(input_array)
        output_tensor = torch.from_numpy(output_array)
        model = load_safetensors_model(model_path, device="cpu")
        model.eval()
        ready_queue.put(
            {
                "member_id": member_id,
                "cpu_set": list(cpu_set),
                "threads": torch.get_num_threads(),
            }
        )

        while True:
            barrier.wait(timeout=WORKER_FORWARD_TIMEOUT_SECONDS)
            current_command = command.value
            if current_command == STOP_COMMAND:
                break
            axes = (
                ()
                if current_command == NO_FLIP_COMMAND
                else TTA_FLIP_AXES[current_command]
            )
            with torch.inference_mode():
                model_input = torch.flip(input_tensor, axes) if axes else input_tensor
                probabilities = _logits_to_probabilities(model(model_input))
                if axes:
                    probabilities = torch.flip(probabilities, axes)
                if tuple(probabilities.shape) != tuple(output_tensor.shape):
                    raise RuntimeError(
                        f"{member_id} produced shape {tuple(probabilities.shape)}, "
                        f"expected {tuple(output_tensor.shape)}"
                    )
                output_tensor.copy_(probabilities)
            barrier.wait(timeout=WORKER_FORWARD_TIMEOUT_SECONDS)
    except BaseException as exc:
        error_queue.put(
            {
                "member_id": member_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        try:
            barrier.abort()
        except BaseException:
            pass
    finally:
        del input_array
        del output_array


def _next_worker_error(error_queue, timeout=0):
    try:
        return error_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def _format_worker_error(error) -> str:
    if error is None:
        return "parallel ensemble worker failed without an error report"
    return f"parallel ensemble worker {error['member_id']} failed: {error['error']}"


class ParallelEnsemblePatchInferenceEngine(PatchInferenceEngine):
    """Run one CPU process per model while retaining PatchInferenceEngine I/O."""

    def __init__(
        self,
        model_paths,
        member_ids,
        config,
        *,
        output_channels: int,
        threads_per_model: int = THREADS_PER_MODEL,
        sw_batch_size: int = 1,
    ):
        paths = tuple(Path(path) for path in model_paths)
        members = tuple(member_ids)
        if len(paths) < 2:
            raise ValueError("parallel ensemble requires at least two model paths")
        if len(paths) != len(members):
            raise ValueError("model_paths and member_ids must have equal length")
        if output_channels < 1:
            raise ValueError("output_channels must be positive")
        if sw_batch_size != 1:
            raise ValueError("parallel ensemble currently requires sw_batch_size=1")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"parallel ensemble model files not found: {missing}"
            )

        self.model_paths = paths
        self.member_ids = members
        self.output_channels = output_channels
        self.threads_per_model = threads_per_model
        self.worker_cpu_sets = assign_worker_cpus(
            os.sched_getaffinity(0), len(paths), threads_per_model
        )

        # The parent owns preprocessing, aggregation, and postprocessing. The
        # identity module satisfies the base engine's inference-only setup; its
        # forward method is never called.
        super().__init__(torch.nn.Identity(), config, sw_batch_size=sw_batch_size)

    def _wait_for_workers(self, processes, ready_queue, error_queue):
        ready = []
        deadline = time.monotonic() + WORKER_STARTUP_TIMEOUT_SECONDS
        while len(ready) < len(processes):
            error = _next_worker_error(error_queue)
            if error is not None:
                raise RuntimeError(_format_worker_error(error))
            failed = [
                process for process in processes if process.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(
                    f"parallel ensemble worker exited during startup with code "
                    f"{failed[0].exitcode}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("parallel ensemble workers timed out during startup")
            try:
                ready.append(ready_queue.get(timeout=min(0.2, remaining)))
            except queue.Empty:
                pass
        for worker in sorted(ready, key=lambda item: item["member_id"]):
            cpus = ",".join(str(cpu) for cpu in worker["cpu_set"])
            print(
                f"  Ready {worker['member_id']}: {worker['threads']} threads "
                f"on CPUs {cpus}"
            )

    def _dispatch(self, barrier, command, command_value, error_queue):
        command_value.value = command
        try:
            barrier.wait(timeout=WORKER_FORWARD_TIMEOUT_SECONDS)
            barrier.wait(timeout=WORKER_FORWARD_TIMEOUT_SECONDS)
        except BrokenBarrierError as exc:
            error = _next_worker_error(error_queue, timeout=1)
            raise RuntimeError(_format_worker_error(error)) from exc

    def _stop_workers(self, processes, barrier, command_value):
        alive = [process for process in processes if process.is_alive()]
        if alive:
            command_value.value = STOP_COMMAND
            try:
                barrier.wait(timeout=5)
            except BrokenBarrierError:
                pass
        for process in processes:
            process.join(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
            process.close()

    def _run_inference(self, prepared, tta: bool = False) -> torch.Tensor:
        patch_iterator = iter(prepared.patch_loader)
        try:
            first_batch = next(patch_iterator)
        except StopIteration as exc:
            raise RuntimeError("patch sampler produced no patches") from exc

        first_input = first_batch["image"][tio.DATA]
        if first_input.device.type != "cpu" or first_input.dtype != torch.float32:
            first_input = first_input.to(device="cpu", dtype=torch.float32)
        input_shape = tuple(first_input.shape)
        output_shape = (input_shape[0], self.output_channels, *input_shape[2:])
        member_output_shape = output_shape
        all_output_shape = (len(self.model_paths), *output_shape)
        bytes_per_member = (
            int(np.prod(member_output_shape)) * np.dtype(np.float32).itemsize
        )

        context = mp.get_context("spawn")
        command_value = context.Value("i", NO_FLIP_COMMAND)
        barrier = context.Barrier(len(self.model_paths) + 1)
        ready_queue = context.Queue()
        error_queue = context.Queue()

        with tempfile.TemporaryDirectory(prefix="vs-parallel-ensemble-") as directory:
            buffer_dir = Path(directory)
            input_path = buffer_dir / "input.float32"
            output_path = buffer_dir / "output.float32"
            input_array = np.memmap(
                input_path, dtype=np.float32, mode="w+", shape=input_shape
            )
            output_array = np.memmap(
                output_path, dtype=np.float32, mode="w+", shape=all_output_shape
            )
            input_tensor = torch.from_numpy(input_array)
            output_tensor = torch.from_numpy(output_array)

            processes = [
                context.Process(
                    name=f"vs-ensemble-{member_id}",
                    target=_ensemble_worker,
                    args=(
                        member_id,
                        str(model_path),
                        cpu_set,
                        str(input_path),
                        input_shape,
                        str(output_path),
                        member_output_shape,
                        index * bytes_per_member,
                        command_value,
                        barrier,
                        ready_queue,
                        error_queue,
                    ),
                )
                for index, (member_id, model_path, cpu_set) in enumerate(
                    zip(self.member_ids, self.model_paths, self.worker_cpu_sets)
                )
            ]
            for process in processes:
                process.start()

            try:
                self._wait_for_workers(processes, ready_queue, error_queue)
                flip_commands = range(len(TTA_FLIP_AXES)) if tta else (NO_FLIP_COMMAND,)
                for patches_batch in itertools.chain((first_batch,), patch_iterator):
                    patch_input = patches_batch["image"][tio.DATA]
                    patch_input = patch_input.to(device="cpu", dtype=torch.float32)
                    if tuple(patch_input.shape) != input_shape:
                        raise RuntimeError(
                            f"parallel ensemble patch shape changed from {input_shape} "
                            f"to {tuple(patch_input.shape)}"
                        )
                    input_tensor.copy_(patch_input)
                    summed_probabilities = None
                    for command in flip_commands:
                        self._dispatch(barrier, command, command_value, error_queue)
                        for member_index in range(len(self.model_paths)):
                            member_probabilities = output_tensor[member_index]
                            summed_probabilities = (
                                member_probabilities.clone()
                                if summed_probabilities is None
                                else summed_probabilities + member_probabilities
                            )
                    divisor = len(self.model_paths) * (len(TTA_FLIP_AXES) if tta else 1)
                    probabilities = summed_probabilities / divisor
                    prepared.aggregator.add_batch(
                        probabilities, patches_batch[tio.LOCATION]
                    )
            finally:
                self._stop_workers(processes, barrier, command_value)
                del input_tensor
                del output_tensor
                del input_array
                del output_array

        return prepared.aggregator.get_output_tensor()
