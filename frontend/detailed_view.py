"""Lossless, bounded scan transport for the offline browser volume viewer.

Only scan/frame changes transfer voxels. Camera, slices and contrast are owned
by the component, so they neither rerun Streamlit nor invoke a detector/VTK.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import zlib

import numpy as np

from core import ScanCase


MAX_VOXELS = 64 * 1024 * 1024
MAX_TRANSFER_BYTES = 128 * 1024 * 1024


def scan_revision(image: Path, mask: Path | None, image_mtime: int,
                  mask_mtime: int | None, frame: int) -> str:
    """A revision separates frames, files, and edits, including same-named cases."""
    identity = (str(image.resolve()), str(mask.resolve()) if mask else None,
                image_mtime, mask_mtime, frame)
    return hashlib.sha256(repr(identity).encode()).hexdigest()[:24]


def _compress(array: np.ndarray, *, mask: bool = False) -> bytes:
    compressor = zlib.compressobj(level=1, wbits=31)  # gzip; browser built-in decoder
    chunks = []
    # X varies fastest on the wire. Slabs avoid a second full-volume copy.
    for k in range(array.shape[2]):
        slab = (array[:, :, k] > 0).astype('u1') if mask else array[:, :, k].astype('<f4')
        chunks.append(compressor.compress(slab.tobytes(order='F')))
    chunks.append(compressor.flush())
    return b''.join(chunks)


def volume_payload(case: ScanCase, *, frame: int = 0, revision: str = "array") -> dict:
    """Preserve float32 values, native indices and the complete NIfTI affine.

    These are display coordinates, not new evaluator measurements. Non-mm
    headers are converted for overlay; unknown units cannot safely host mm
    detector markers and are explicitly identified.
    """
    volume = case.image.volume(frame)
    if volume.size > MAX_VOXELS:
        raise ValueError("Interactive view supports at most 64 million voxels per frame. "
                         "Use the VTK snapshot or desktop viewer for this scan.")
    geometry = case.image.geometry
    factor = {'mm': 1., 'meter': 1000., 'micron': .001}.get(geometry.spatial_unit)
    affine = geometry.affine_ras.copy()
    affine[:3] *= factor if factor is not None else 1.
    shape = volume.shape
    corners = np.array(np.meshgrid(*[(-.5, n - .5) for n in shape], indexing='ij')).reshape(3, -1).T
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    mask_bounds = None
    mask_data = b''
    if case.mask is not None:
        parent = case.mask.volume(frame if case.mask.data.ndim == 4 else 0)
        occupied = [np.flatnonzero(np.any(parent > 0, axis=tuple(a for a in range(3) if a != axis)))
                    for axis in range(3)]
        if all(len(axis) for axis in occupied):
            mask_bounds = [[float(axis[0]) - .5, float(axis[-1]) + .5] for axis in occupied]
        mask_data = _compress(parent, mask=True)
    ct_data = _compress(volume)
    if len(ct_data) + len(mask_data) > MAX_TRANSFER_BYTES:
        raise ValueError("Compressed scan exceeds the 128 MiB interactive transfer limit. "
                         "Use the VTK snapshot or desktop viewer.")
    return {'meta': {'id': revision, 'case_id': case.case_id, 'shape': list(shape),
                     'affine': affine.tolist(), 'inverse': np.linalg.inv(affine).tolist(),
                     'spacing': np.linalg.norm(affine[:3, :3], axis=0).tolist(),
                     'axis_codes': list(geometry.axis_codes),
                     'unit': 'mm' if factor is not None else geometry.spatial_unit,
                     'markers_supported': factor is not None,
                     'center': world.mean(axis=0).tolist(),
                     'span': max(float(np.linalg.norm(np.ptp(world, axis=0))), 1.),
                     'corners': world.tolist(), 'mask_bounds': mask_bounds,
                     'has_mask': case.mask is not None,
                     'transfer_bytes': len(ct_data) + len(mask_data),
                     'decoded_bytes': volume.size * (5 if case.mask is not None else 4)},
            'ct_gzip': ct_data, 'mask_gzip': mask_data}


@lru_cache(maxsize=1)
def _component():
    import streamlit.components.v1 as components
    return components.declare_component('detailed_volume',
                                        path=str(Path(__file__).with_name('detailed_component')))


def detailed_view(payload: dict, *, prediction: dict | None, selected_branch: str | None,
                  show_branches: bool, key: str = 'detailed_volume') -> None:
    import streamlit as st

    previous = st.session_state.get(key)
    # Acknowledgment is tied to this volume revision. Remounts request it again;
    # ordinary branch/model updates carry only metadata and small marker lists.
    loaded = (isinstance(previous, dict) and previous.get('id') == payload['meta']['id']
              and previous.get('event') == 'loaded')
    _component()(meta=payload['meta'],
                 ct_gzip=None if loaded else payload['ct_gzip'],
                 mask_gzip=None if loaded else payload['mask_gzip'],
                 branches=(prediction or {}).get('daughters', []),
                 selected_branch=selected_branch, show_branches=show_branches,
                 key=key, default=None)
