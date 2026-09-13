# Interactive viewer measurements

Recorded on 2026-09-13 using subject001 (512 × 512 × 174), Windows,
Python 3.13.7, and Node.js 22.21.0. Raw samples are preserved in
[viewer_20260913.json](experiments/viewer_20260913.json).

The former Detailed View required a Streamlit rerun, a CPU VTK render, and a
PNG response for each control change. Detailed View now transfers scan buffers
once and renders in a browser worker. VTK Snapshot retains the previous renderer.
No detector implementation, parameter, prediction, or evaluation result is changed
by this viewer work.

| Work | Measured time |
| --- | ---: |
| Previous VTK warm camera update, 1000 × 760 PNG, median of 5 | 398 ms |
| New worker, aorta + planes, 96-pixel preview, median of 5 | 3.8 ms |
| New worker, aorta + planes, 280-pixel refinement, median of 5 | 15.1 ms |
| New worker, CT + aorta + planes, 96-pixel preview, median of 5 | 10.1 ms |
| New worker, CT + aorta + planes, 280-pixel refinement, median of 5 | 83.1 ms |
| Three full-resolution native slices, initial computation | 31.0 ms |
| Initial lossless compression in Python | 2.25 s |
| Initial decompression in the worker harness | 0.82 s |

These are local work timings, **not measured browser FPS or end-to-end latency**.
The Node harness executes the actual worker and rendering code, including
cooperative scheduling. It excludes Streamlit transfer, browser message copying,
canvas painting and layout. VTK and the new adaptive renderer also differ in
resolution and rendering algorithm; this is a responsiveness/quality tradeoff,
not an equal-quality renderer speed comparison. The 280-pixel refinement starts
220 ms after the last gesture. All 2-D slice data retain native resolution and
scaled float32 values; no display downsampling changes the source scans.

Subject001 transfers 55.1 MiB of compressed data and retains 217.5 MiB of decoded
voxel buffers in the worker. Python scan buffers, compression, transport objects,
and canvas images add memory beyond that figure. Loading a new case takes longer
than subsequent interaction. Case/frame changes reload data; camera, slices,
windowing and local branch inspection send **zero** Streamlit widget updates.
An initial loaded acknowledgment lets subsequent model/marker updates omit voxels.

Reproduce the benchmark from the repository root with Node.js installed as
optional development tooling:

```powershell
python scripts/benchmark_detailed_view.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --vtk --output tmp/viewer_benchmark.json
```

Omit `--vtk` to measure only the new worker. The benchmark uses temporary local
files and no external services. It is separate from the offline evaluator.

Verification includes lossless CT/mask round trips, frame and file revision
invalidation, physical coordinates checked against SimpleITK, native slice
pixel addressing, CPU volume/plane rendering, thresholds, empty masks, bounded
payloads, and the actual worker/component message flow under rapid input.
The Streamlit app is exercised through its test runner, and its local server and
bundled component assets are checked offline. Generated subject001 render images
were inspected for slice/volume alignment. No controllable browser was available
in this session; actual browser appearance, gestures, and PNG downloads still
need a manual browser check. The interactive viewer displays per-frame timings
to help assess performance on the user's machine.
