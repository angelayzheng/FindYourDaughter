"""Prepare CT review evidence without converting model proposals into truth."""

from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import re
import shutil

import numpy as np
import SimpleITK as sitk

from backend.configuration import configuration
from backend.detectors import detect
from backend.inputs import CaseData, load_case, _same_geometry
from evaluation.draft_set import _sha256, _write_json, read_reference


CHECKLIST = [
    'Inspect the full circumference of the entire supplied parent segment for missed origins.',
    'Confirm continuous CT lumen from the parent into each proposed direct daughter in consecutive orthogonal slices.',
    'Separate nearby ostia; one common trunk is one origin. Do not substitute a downstream daughter.',
    'Exclude superior/inferior crop faces and the terminal iliac division.',
    'Measure origin diameter separately from seed radius; record uncertainty at the provisional 2 mm cutoff.',
    'Trace at least 5 mm beyond the origin; stop at 10 mm or the first bifurcation. Resolve early forks explicitly.',
    'Check each seed in the daughter lumen, physical LPS coordinates, direction, mask width and parent exclusion.',
    'Record unobservable regions, exclusions, completeness, reviewer identity and unresolved decisions.',
]


def physical_affine(image: sitk.Image) -> np.ndarray:
    """Index XYZ -> SimpleITK LPS mm, constructed using the required transform."""
    affine = np.eye(4)
    origin = np.array(image.TransformIndexToPhysicalPoint((0, 0, 0)))
    affine[:3, 3] = origin
    for axis in range(3):
        index = tuple(int(i == axis) for i in range(3))
        affine[:3, axis] = np.array(image.TransformIndexToPhysicalPoint(index)) - origin
    return affine


def _viewer(case, review: dict, output: Path) -> None:
    """Self-contained offline tri-planar CT viewer; no server or CDN."""
    ct = sitk.GetArrayViewFromImage(case.image)
    parent = sitk.GetArrayViewFromImage(case.aorta_mask)
    spacing = np.array(case.image.GetSpacing())[::-1]
    bounds = []
    for axis in range(3):
        active = np.flatnonzero(np.any(parent, axis=tuple(i for i in range(3) if i != axis)))
        pad = int(np.ceil(25 / spacing[axis]))
        bounds.append((max(0, int(active[0]) - pad), min(ct.shape[axis], int(active[-1]) + pad + 1)))
    selection = tuple(slice(lo, hi) for lo, hi in bounds)
    values = np.asarray(ct[selection], dtype=np.float32)
    local_parent = np.asarray(parent[selection] > 0, dtype=np.uint8)
    upper = max(450., float(np.percentile(values[local_parent > 0], 90)) + 150)
    # Keep 16-bit HU values in the review viewer so window/level remain adjustable.
    encoded = np.clip(np.rint(values), -32768, 32767).astype('<i2')
    payload = {**review, 'crop_offset_xyz': [lo for lo, hi in bounds][::-1],
               'crop_size_xyz': list(values.shape[::-1]), 'display_window': upper + 100,
               'display_level': (upper - 100) / 2,
               'ct_base64': base64.b64encode(encoded.tobytes()).decode(),
               'parent_base64': base64.b64encode(local_parent.tobytes()).decode()}
    template = Path(__file__).with_name('annotation_viewer.html').read_text(encoding='utf-8')
    data = json.dumps(payload, allow_nan=False).replace('<', '\\u003c')
    output.write_text(template.replace('__REVIEW_DATA__', data), encoding='utf-8')


def prepare_workspace(dataset: Path, output: Path, *, existing_draft: Path | None = None,
                      hint_configs: list[dict] | None = None, expected_cases: int | None = 25) -> dict:
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset):
        raise ValueError('Review output must be outside the source dataset')
    directories = sorted(p for p in dataset.glob('subject*') if p.is_dir())
    if not directories or (expected_cases is not None and len(directories) != expected_cases):
        raise ValueError(f'Expected {expected_cases} source cases, found {len(directories)}')
    inputs = []
    for directory in directories:
        match = re.fullmatch(r'subject(\d+)', directory.name)
        if not match:
            raise ValueError(f'Invalid subject directory {directory.name}')
        number = int(match[1])
        images, masks = list(directory.glob('orig*.nii*')), list(directory.glob('mask*.nii*'))
        if len(images) != 1 or len(masks) != 1:
            raise ValueError(f'Expected one CT and one parent mask in {directory}')
        inputs.append((number, images[0], masks[0]))
    if len({n for n, _, _ in inputs}) != len(inputs):
        raise ValueError('Duplicate numeric case IDs')
    configs = [] if hint_configs is None else [configuration(c['detector'], c['parameters']) for c in hint_configs]
    existing_draft = Path(existing_draft).resolve() if existing_draft else None
    if existing_draft and output.is_relative_to(existing_draft):
        raise ValueError('Review output must be outside existing annotations')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Annotation review workspace, not a labeled ground-truth dataset.',
                'annotation_status': 'unreviewed', 'comparison_ready': False,
                'coordinate_system': 'SimpleITK physical LPS millimetres',
                'policy': {'minimum_origin_diameter_mm': 2., 'diameter_rule_status': 'provisional policy inherited from the five-case draft; challenge PDF defers the final cutoff',
                           'minimum_visible_path_mm': 5., 'maximum_trace_mm': 10., 'stop_at_first_bifurcation': True},
                'hint_configurations': configs, 'backend_sha256': {p.name: _sha256(p) for p in sorted((Path(__file__).resolve().parents[1] / 'backend').glob('*.py'))},
                'cases': [], 'checklist': CHECKLIST}
    _write_json(output / 'manifest.json', manifest)
    if existing_draft:
        # Copy, never relabel or modify, the original five-case draft release.
        shutil.copytree(existing_draft, output / 'existing_draft')
    for number, image_path, mask_path in inputs:
        folder = output / f'case_{number}'
        folder.mkdir()
        source_hashes = {str(p): _sha256(p) for p in (image_path, mask_path)}
        case = load_case(image_path, mask_path)
        mask_values = sitk.GetArrayViewFromImage(case.aorta_mask)
        if not np.all((mask_values == 0) | (mask_values == 1)) or not np.any(mask_values):
            raise ValueError(f'{image_path.parent.name}: requires a nonempty binary parent mask')
        resampled = case.image.HasMetaDataKey('branchseed_geometry')
        saved = []
        corner_errors = []
        for name, volume in (('ct_working.nii.gz', case.image), ('aorta_working.nii.gz', case.aorta_mask)):
            sitk.WriteImage(volume, str(folder / name), True)
            reopened = sitk.ReadImage(str(folder / name))
            corners = np.array([(*p, 1) for p in product(*[(0, n-1) for n in volume.GetSize()])])
            drift = float(np.linalg.norm((corners @ (physical_affine(volume)-physical_affine(reopened)).T)[:, :3], axis=1).max())
            if volume.GetSize() != reopened.GetSize() or drift > .001 or not np.array_equal(sitk.GetArrayViewFromImage(volume), sitk.GetArrayViewFromImage(reopened)):
                raise ValueError(f'{number}: saved working image failed round-trip validation')
            corner_errors.append(drift)
            saved.append(reopened)
        if not _same_geometry(*saved):
            raise ValueError(f'{number}: saved CT and parent grids differ')
        # NIfTI-1 headers store float32 transforms. Derive annotation coordinates
        # from the reopened, paired working grid instead of mixing both versions.
        case = CaseData(image_path, mask_path, saved[0], saved[1])
        old = existing_draft / f'case_{number}' if existing_draft else None
        legacy_count = None
        if old and old.is_dir():
            reference = read_reference(old)
            if (_sha256(old / f'orig{number}.nii.gz') != source_hashes[str(image_path)]
                    or _sha256(old / f'aorta{number}.nii.gz') != source_hashes[str(mask_path)]):
                raise ValueError(f'Case {number}: existing annotations refer to different source files')
            legacy_count = len(reference.annotation['daughters'])
        review = {'case_id': f'case_{number}', 'subject_id': image_path.parent.name,
                  'annotation_status': 'unreviewed', 'comparison_ready': False,
                  'daughters': None, 'parent': {'instance_id': 'aorta'},
                  'coordinate_system': manifest['coordinate_system'],
                  'shape_xyz': list(case.image.GetSize()), 'spacing_xyz_mm': list(case.image.GetSpacing()),
                  'working_index_to_lps_mm': physical_affine(case.image).tolist(),
                  'working_lps_mm_to_index': np.linalg.inv(physical_affine(case.image)).tolist(),
                  'maximum_header_roundtrip_corner_error_mm': max(corner_errors),
                  'resampled_nonorthogonal_source': resampled, 'source_sha256': source_hashes,
                  'existing_draft_instances': legacy_count,
                  'reference_status': 'existing_draft_needs_expert_review' if legacy_count is not None else 'no_reference_annotations',
                  'full_parent_review_complete': False, 'reviewer': None, 'review_date': None,
                  'notes': [], 'manual_guides': [], 'proposals': [], 'hint_runs': [], 'checklist': CHECKLIST}
        for index, config in enumerate(configs, 1):
            result = detect(case.image, case.aorta_mask, detector=config['detector'], parameters=config['parameters'])
            run_id = f'hint_{index}_{config["detector"]}'
            _write_json(folder / 'model_hints' / f'{run_id}.json',
                        {'configuration': config, 'daughters': result.daughters(), 'diagnostics': result.diagnostics})
            review['hint_runs'].append({'run_id': run_id, 'count': len(result.branches), 'configuration': config})
            for branch in result.branches:
                # Exact duplicate geometry is one review card, never an anatomical merge.
                existing = next((p for p in review['proposals'] if p['geometry'] == asdict(branch)), None)
                if existing:
                    existing['sources'].append(run_id)
                    continue
                review['proposals'].append({'proposal_id': f'proposal_{len(review["proposals"])+1:03d}',
                                            'sources': [run_id], 'decision': 'unreviewed', 'review_notes': '',
                                            'geometry': asdict(branch)})
        _write_json(folder / 'annotations.json', review)
        _viewer(case, review, folder / 'review.html')
        if any(_sha256(Path(path)) != digest for path, digest in source_hashes.items()):
            raise ValueError(f'Case {number}: source changed while preparing review')
        record = {key: review[key] for key in ('case_id', 'subject_id', 'annotation_status', 'reference_status',
                                              'comparison_ready', 'existing_draft_instances', 'shape_xyz',
                                              'spacing_xyz_mm', 'resampled_nonorthogonal_source', 'source_sha256')}
        record['maximum_header_roundtrip_corner_error_mm'] = max(corner_errors)
        record['proposal_cards'] = len(review['proposals'])
        record['working_sha256'] = {p.name: _sha256(p) for p in folder.glob('*working.nii.gz')}
        manifest['cases'].append(record)
        _write_json(output / 'manifest.json', manifest)
        print(f'{record["subject_id"]}: {legacy_count if legacy_count is not None else "no"} existing draft labels; '
              f'{record["proposal_cards"]} unreviewed proposal cards; source/grid checks passed', flush=True)
    manifest['status'] = 'complete_workspace_annotations_pending'
    _write_json(output / 'manifest.json', manifest)
    rows = ''.join(f'<tr><td><a href="{c["case_id"]}/review.html">{c["subject_id"]}</a></td>'
                   f'<td>{c["existing_draft_instances"] if c["existing_draft_instances"] is not None else "Unknown"}</td>'
                   f'<td>{c["proposal_cards"]}</td><td>Not ready</td></tr>' for c in manifest['cases'])
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>25-case annotation review</title>'
        '<style>body{font:18px system-ui;max-width:1000px;margin:40px auto;color:#172330}td,th{padding:8px 24px;text-align:left}</style>'
        '<h1>Annotation review workspace</h1><p>No case has been certified as ground truth. Model proposals are hints, '
        'not reference labels or verified branch counts. Unknown does not mean zero daughters.</p><p>The five existing '
        'drafts are preserved under <code>existing_draft/</code>. Open a case for full-resolution tri-planar review; '
        'model hints start hidden. Original NIfTI files remain unchanged.</p><table><tr><th>Case</th><th>Existing draft labels</th>'
        f'<th>Proposal cards</th><th>Scoring status</th></tr>{rows}</table></html>', encoding='utf-8')
    return manifest
