import base64
import json
from functools import lru_cache
import struct
import zlib
from pathlib import Path

import numpy as np

from .paths import PROJECT


def png(array):
    pixels = np.clip(array * 255, 0, 255).astype(np.uint8)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    height, width = pixels.shape[:2]
    def chunk(kind, content):
        return struct.pack("!I", len(content)) + kind + content + struct.pack("!I", zlib.crc32(kind + content) & 0xffffffff)
    raw = b"".join(b"\x00" + row.tobytes() for row in pixels)
    value = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0)) +
             chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(value).decode()


@lru_cache(maxsize=2)
def volume_arrays(path, modified):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key][0, 0].copy() for key in ("input", "mask", "extracted", "warped")}


def volume_slice(folder, branch, stage, index):
    if branch not in {"baseline", "candidate"} or stage not in {"input", "mask", "extracted", "warped"}:
        raise ValueError("Unknown volume")
    path = folder / f"{branch}_preview.npz"
    volume = volume_arrays(str(path), path.stat().st_mtime_ns)[stage]
    if type(index) is not int or not 0 <= index < min(volume.shape):
        raise ValueError("Slice index outside the volume")
    return {"index": index, "shape": list(volume.shape),
            "planes": [png(np.rot90(np.take(volume, index, axis=axis))) for axis in range(3)]}


def previews(folder, full=False):
    result = {}
    for name in ("baseline", "candidate"):
        if not (folder / f"{name}_preview.npz").exists():
            continue
        with np.load(folder / f"{name}_preview.npz", allow_pickle=False) as arrays:
            images = {}
            for key in ("input", "mask", "extracted", "warped"):
                volume = arrays[key][0, 0]
                images[key] = png(np.rot90(volume[:, :, 48]))
            images["shape"] = list(arrays["input"].shape[2:])
            images["run_id"] = folder.name
            images["slice_indices"] = list(range(min(images["shape"]))) if full else [48]
            images["planes"] = {}
            for key in ("input", "mask", "extracted", "warped"):
                volume = arrays[key][0, 0]
                images["planes"][key] = [[png(np.rot90(np.take(volume, index, axis=axis)))
                                          for index in images["slice_indices"]] for axis in range(3)]
            adjacency = np.clip((arrays["edges"][0] + 1) / 2, 0, 1)
            images["edges"] = png(np.stack((0.1 + 0.12 * adjacency, 0.15 + 0.65 * adjacency, 0.2 + 0.45 * adjacency), -1))
            images["graph_values"] = arrays["edges"][0].round(5).tolist()
            result[name] = images
    return result


from .frontend import document


def render(report, folder):
    payload = {**report, "previews": previews(folder, full=True)}
    (folder / "index.html").write_text(document(payload), encoding="utf-8")


def landing():
    return document(None)
