"""Measure local viewer work; Node timings are not browser FPS measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import ScanCase, VolumeViewOptions
from frontend.detailed_view import volume_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--aorta-mask', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vtk', action='store_true', help='Also measure the previous server PNG path')
    args = parser.parse_args()
    node = shutil.which('node')
    if node is None:
        parser.error('Install Node.js on the development machine to run this benchmark.')
    started = perf_counter()
    case = ScanCase.from_nifti(args.image, args.aorta_mask)
    load_ms = (perf_counter()-started)*1000
    started = perf_counter()
    payload = volume_payload(case, revision=case.case_id)
    report = {'case_id': case.case_id, 'shape': list(case.image.shape),
              'python': platform.python_version(), 'platform': platform.system(),
              'node': subprocess.check_output([node, '--version'], text=True).strip(),
              'scan_load_ms': load_ms, 'pack_ms': (perf_counter()-started)*1000,
              'transfer_bytes': payload['meta']['transfer_bytes'],
              'decoded_voxel_bytes': payload['meta']['decoded_bytes'],
              'limitations': 'Node runs the actual CPU worker logic. Timings exclude a browser DOM, '
                            'canvas painting, Streamlit transport and network. The adaptive preview '
                            'and VTK PNG differ in resolution and rendering algorithm.'}
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root/'meta.json').write_text(json.dumps(payload['meta']), encoding='utf-8')
        (root/'ct.gz').write_bytes(payload['ct_gzip'])
        (root/'mask.gz').write_bytes(payload['mask_gzip'])
        result = subprocess.run([node, str(Path(__file__).with_suffix('.cjs')), str(root)],
                                capture_output=True, text=True, timeout=120, check=True)
        report['javascript'] = json.loads(result.stdout)
    if args.vtk:
        from frontend.render import VTKRenderSession, render_3d
        options = VolumeViewOptions(max_dimension=128, window=600, level=200)
        started = perf_counter()
        session = VTKRenderSession(args.image, args.aorta_mask, options)
        report['vtk_session_start_ms'] = (perf_counter()-started)*1000
        try:
            samples = []
            for i in range(6):
                started = perf_counter()
                render_3d(case, options, azimuth=30+i*5, elevation=25,
                          show_volume=False, show_mask=True, show_slices=True,
                          show_planes=True, session=session)
                elapsed = (perf_counter()-started)*1000
                if i == 0:
                    report['vtk_first_frame_ms'] = elapsed
                else:
                    samples.append(elapsed)
            report['vtk_warm_camera_ms'] = {'samples': samples, 'median': statistics.median(samples)}
        finally:
            session.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
