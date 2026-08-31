"""Execution-device resolution for ordinary and CPU-offloaded tensors."""

from __future__ import annotations

import torch


def tensor_compute_device(tensor: torch.Tensor) -> torch.device:
    """Return the device where operations using ``tensor`` will execute.

    FSDP2 ``CPUOffloadPolicy`` keeps a DTensor's local shard on CPU between
    forwards while its device mesh remains accelerator-backed.  A child
    FSDP pre-hook materializes the full tensor on the current rank accelerator,
    so inputs must target the mesh device rather than the shard's physical
    storage device.  Ordinary tensors retain their physical device.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("compute-device resolution requires a tensor")
    physical_device = tensor.device
    device_mesh = getattr(tensor, "device_mesh", None)
    mesh_device_type = getattr(device_mesh, "device_type", None)
    if physical_device.type != "cpu" or mesh_device_type in {None, "cpu"}:
        return physical_device
    if not isinstance(mesh_device_type, str) or not mesh_device_type:
        raise TypeError("tensor device mesh has an invalid device type")
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None or accelerator.type != mesh_device_type:
        actual = None if accelerator is None else accelerator.type
        raise RuntimeError(
            "CPU-offloaded tensor mesh differs from the current accelerator: "
            f"mesh={mesh_device_type!r}, accelerator={actual!r}"
        )
    return torch.device(mesh_device_type, torch.accelerator.current_device_index())


__all__ = ["tensor_compute_device"]
