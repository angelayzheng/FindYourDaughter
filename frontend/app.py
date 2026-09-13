"""Local, CPU-only inspection dashboard for CT scans and aorta branches."""

from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import json
import sys
from time import monotonic, sleep

import streamlit as st
import streamlit.components.v1 as components

# Streamlit runs this file as a script, so make repository packages importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.detectors import DETECTOR_NAMES
from backend.pipeline import run_case
from core import ScanCase, VolumeViewOptions
from frontend.browser_scene import scene_html, scene_payload
from frontend.cases import find_subjects, mask_choices
from frontend.detailed_view import detailed_view, scan_revision, volume_payload
from frontend.point_cloud import render_projection
from frontend.render import VTKRenderSession, VTKUnavailable, render_3d
from frontend.results import comparison_csv, discover_csv_files, read_csv_file, synthetic_comparison_rows


@st.cache_resource(show_spinner=False, max_entries=1)
def load_case(
    image: Path, mask: Path | None, image_mtime: int, mask_mtime: int | None
) -> ScanCase:
    """Reuse loaded tensors across camera and display changes."""
    return ScanCase.from_nifti(image, mask)


@st.cache_data(show_spinner=False, max_entries=32)
def load_prediction(
    image: Path, mask: Path, image_mtime: int, mask_mtime: int, detector: str
) -> dict:
    """Run the evaluator detector once per input revision and algorithm."""
    return run_case(image, mask, detector=detector)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_browser_volume(image: Path, mask: Path | None, image_mtime: int,
                        mask_mtime: int | None, frame: int) -> dict:
    """Compress one current frame once; camera/contrast are browser-local."""
    case = load_case(image, mask, image_mtime, mask_mtime)
    return volume_payload(case, frame=frame,
                          revision=scan_revision(image, mask, image_mtime, mask_mtime, frame))


def prepare_view(
    case: ScanCase,
    options: VolumeViewOptions,
    *,
    azimuth: int,
    elevation: int,
    zoom: float,
    show_volume: bool,
    show_mask: bool,
    mip: bool,
    cut_at_k: int | None,
    show_slices: bool,
    show_planes: bool,
    plane_opacity: float,
    focus_mask: bool,
    slice_indices: tuple[int, int, int],
    branches: list[dict],
    show_branches: bool,
    selected_branch: str | None,
    session: VTKRenderSession | None,
) -> tuple[bytes, str, str, tuple[int, int] | None, VTKRenderSession | None]:
    """Render the native VTK scene or a safe CPU projection fallback."""
    try:
        if session is None:
            session = VTKRenderSession(
                case.image.source_path,
                case.mask.source_path if case.mask else None,
                options,
            )
        png = render_3d(
            case,
            options,
            azimuth=azimuth,
            elevation=elevation,
            zoom=zoom,
            show_volume=show_volume,
            show_mask=show_mask,
            mip=mip,
            cut_at_k=cut_at_k,
            show_slices=show_slices,
            show_planes=show_planes,
            plane_opacity=plane_opacity,
            focus_mask=focus_mask,
            slice_indices=slice_indices,
            branches=branches,
            show_branches=show_branches,
            selected_branch=selected_branch,
            session=session,
        )
        return png, "VTK volume", "", None, session
    except VTKUnavailable as error:
        if session is not None:
            session.close()
        png, ct_count, mask_count = render_projection(
            case,
            frame=options.frame,
            window=options.window,
            level=options.level,
            show_volume=show_volume,
            show_mask=show_mask,
            azimuth=azimuth,
            elevation=elevation,
            branches=branches,
            show_branches=show_branches,
            selected_branch=selected_branch,
        )
        return png, "CPU preview", str(error), (ct_count, mask_count), None


st.set_page_config(page_title="Find Your Daughter", page_icon="🫀", layout="wide")
st.markdown(
    """<style>
  .stApp { background: #171417; color: #f0e7e9; }
  [data-testid="stSidebar"] { background: #211b20; border-right: 1px solid #75515f; }
  [data-testid="stMainBlockContainer"] { padding-top: 3.5rem !important; padding-bottom: 2rem !important; }
  [data-testid="stSidebarHeader"] { height: 2.25rem !important; margin-bottom: 0 !important; }
  [data-testid="stSidebarUserContent"] { padding-top: 0 !important; }
  .stApp, .stApp button, .stApp input, .stApp textarea { font-family: Consolas, 'Courier New', monospace; }
  [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #392b33 !important;
    border-color: #9b6d7d !important;
  }
  [data-baseweb="tooltip"]:has([data-testid="stTooltipContent"]),
  [data-baseweb="tooltip"]:has([data-testid="stTooltipContent"]) > div,
  [data-testid="stTooltipContent"] {
    background-color: #513a46 !important;
    color: #fff1f5 !important;
    border-radius: 6px !important;
  }
  [data-testid="stTooltipContent"] * { color: #fff1f5 !important; }
  [data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"],
  [data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] * {
    cursor: pointer !important;
  }
  [data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"]:hover {
    color: #fff1f5 !important;
  }
  [data-testid="stMarkdownContainer"] :not(pre) > code {
    color: #82c9e3 !important;
  }
  h1, h2, h3 { font-family: Consolas, 'Courier New', monospace !important; letter-spacing: .015em; color: #f6edef !important; }
  h1 { font-size: 2.2rem !important; }
  [data-testid="stExpander"] { border: 1px solid #75515f; border-radius: 0; background: #272027; }
  [data-testid="stMainBlockContainer"] [data-testid="stImage"] {
    width: 100% !important;
    max-width: 100% !important;
  }
  [data-testid="stMainBlockContainer"] [data-testid="stImage"] img {
    display: block;
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
    border: 1px solid #75515f;
    border-radius: 0;
  }
  [data-testid="stCaptionContainer"] { color: #d6bdc5 !important; }
  [data-testid="stCodeBlock"] { border: 1px solid #75515f; border-radius: 0; }
  [data-testid="stDownloadButton"] button {
    border: 1px solid #8a5d6d !important;
    border-radius: 0 !important;
    background: #30232a !important;
    color: #f5e9ed !important;
    padding: 7px 10px !important;
    min-height: 0 !important;
    font: 12px Consolas, 'Courier New', monospace !important;
  }
  [data-testid="stDownloadButton"] button:hover {
    background: #49303b !important;
    color: #f5e9ed !important;
    border-color: #8a5d6d !important;
  }
</style>""",
    unsafe_allow_html=True,
)

panel = st.segmented_control(
    "Panel",
    ["Simple View", "Detailed View", "Results", "Benchmark Results"],
    default="Simple View",
    label_visibility="collapsed",
    width="stretch",
    key="main_panel",
)

with st.sidebar:
    st.image(ROOT / "images" / "FindYourDaughter-Logo-Transparent.png", width=240)
    st.caption("CT review · daughter branches from the aorta")
    with st.expander("01 · CASE FILES", expanded=True):
        dataset_text = st.text_input(
            "Dataset folder",
            value=str(ROOT / "dataset"),
            help="Searches this folder and its subject subfolders for NIfTI scans.",
        )
        dataset = Path(dataset_text).expanduser()
        subjects = find_subjects(dataset)
        if not subjects:
            st.info("No NIfTI scans were found here. Check the dataset folder.")
            if panel != "Benchmark Results":
                st.stop()
        if not subjects and panel == "Benchmark Results":
            st.caption("Benchmark Results does not require a scan selection.")
        if subjects:
            subject = st.selectbox(
                "Subject", list(subjects), key=f"subject:{dataset.resolve()}"
            )
            image = st.selectbox(
                "CT image",
                subjects[subject],
                format_func=lambda path: path.name,
                key=f"image:{dataset.resolve()}:{subject}",
            )
            masks = [None, *mask_choices(image)]
            mask = st.selectbox(
                "Aorta mask",
                masks,
                format_func=lambda path: "None" if path is None else path.name,
                index=1 if len(masks) == 2 else 0,
                key=f"mask:{image.resolve()}",
            )

if panel == "Benchmark Results":
    st.title("Benchmark Results")
    st.subheader("Synthetic evaluation")
    synthetic_root = Path(
        st.text_input(
            "Synthetic dataset folder",
            str(ROOT / "synthetic_dataset"),
            help="Folder containing subject*/orig*.nii, mask*.nii, and truth*.json.",
        )
    ).expanduser()
    synthetic_detector = st.selectbox("Synthetic detector", DETECTOR_NAMES, key="synthetic_detector")
    synthetic_tolerance = st.number_input(
        "Ostium matching tolerance (mm)", min_value=0.1, value=3.0, step=0.5, key="synthetic_tolerance"
    )
    if st.button("Run synthetic evaluation", type="primary"):
        try:
            from scripts.evaluate_synthetic_detection import evaluate

            with st.spinner("Running detector against synthetic truth…"):
                st.session_state["synthetic_report"] = evaluate(
                    synthetic_root, synthetic_tolerance, detector=synthetic_detector
                )
            st.session_state["synthetic_report_error"] = None
        except (OSError, ValueError, RuntimeError) as error:
            st.session_state["synthetic_report"] = None
            st.session_state["synthetic_report_error"] = str(error)
    if st.session_state.get("synthetic_report_error"):
        st.error(st.session_state["synthetic_report_error"])
    report = st.session_state.get("synthetic_report")
    if report:
        metrics = st.columns(5)
        for column, label, value in zip(
            metrics,
            ("True positives", "False positives", "Missed truth", "Ostium error", "Seed error"),
            (
                report["true_positive"],
                report["false_positive"],
                report["false_negative"],
                f"{report['mean_errors'].get('ostium_mm', 0):.2f} mm",
                f"{report['mean_errors'].get('seed_mm', 0):.2f} mm",
            ),
        ):
            column.metric(label, value)
        comparison = synthetic_comparison_rows(report)
        st.caption("True-vs-guess rows use the scorer's one-to-one ostium matching.")
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.download_button(
            "Download true-vs-guess CSV",
            comparison_csv(comparison),
            file_name=f"{synthetic_detector}_synthetic_comparison.csv",
            mime="text/csv",
            on_click="ignore",
        )
        with st.expander("Per-case totals", expanded=False):
            st.dataframe(
                [
                    {key: case[key] for key in ("case_id", "truth", "predictions", "matched")}
                    for case in report["cases"]
                ],
                hide_index=True,
                width="stretch",
            )
    st.divider()
    st.subheader("Saved CSV files")
    results_root = Path(
        st.text_input("Results folder", str(ROOT), help="Recursively search for CSV benchmark outputs.")
    ).expanduser()
    csv_files = discover_csv_files(results_root)
    if not csv_files:
        st.info(f"No CSV result files found under {results_root}.")
    else:
        selected_csv = st.selectbox(
            "CSV result file",
            csv_files,
            format_func=lambda path: str(path.relative_to(results_root.resolve())),
        )
        try:
            columns, rows = read_csv_file(selected_csv)
        except (OSError, UnicodeError, csv.Error) as error:
            st.error(f"Could not read {selected_csv}: {error}")
        else:
            st.caption(f"{len(rows)} rows · {len(columns)} columns")
            if columns:
                st.dataframe(rows, hide_index=True, width="stretch")
                st.download_button(
                    "Download CSV",
                    selected_csv.read_bytes(),
                    file_name=selected_csv.name,
                    mime="text/csv",
                    on_click="ignore",
                )
            else:
                st.warning("The selected CSV has no header row.")
    st.stop()

try:
    image_mtime = image.stat().st_mtime_ns
    mask_mtime = mask.stat().st_mtime_ns if mask else None
    case = load_case(image, mask, image_mtime, mask_mtime)
except (OSError, ValueError, IndexError) as error:
    st.error(f"Could not open the selected scan and mask: {error}")
    st.stop()

shape = case.image.shape
spacing = case.image.geometry.spacing
can_detect = mask is not None and len(shape) == 3
with st.sidebar:
    with st.expander("02 · BRANCH DETECTION", expanded=True):
        detector = st.selectbox(
            "Algorithm",
            DETECTOR_NAMES,
            index=DETECTOR_NAMES.index("refined"),
            help="Refined is the default evaluator algorithm. Baseline and Contact remain available for comparison.",
        )
        if not can_detect:
            st.caption("Select a 3-D CT and its aorta mask to run detection.")
        prediction = None
        detection_error = None
        if can_detect:
            try:
                with st.spinner(f"Detecting candidate branches ({detector})…"):
                    prediction = load_prediction(
                        image, mask, image_mtime, mask_mtime, detector
                    )
            except (OSError, ValueError, RuntimeError) as error:
                detection_error = str(error)
        daughters = prediction["daughters"] if prediction else []
        branch_names = [daughter["instance_id"] for daughter in daughters]
        selected_label = st.selectbox(
            "Highlight branch",
            ["All candidates", *branch_names],
            disabled=not branch_names,
        )
        selected_branch = None if selected_label == "All candidates" else selected_label
        show_branches = st.checkbox(
            "Show branch markers",
            True,
            help="Show the estimated ostium, 5 mm seed, direction arrow, and radius ring.",
            disabled=not branch_names,
        )

# st.title("Find Your Daughter")
st.markdown(
    f"**{case.case_id}** &nbsp; | &nbsp; CT: `{image.name}` &nbsp; | &nbsp; "
    f"{shape[0]} × {shape[1]} × {shape[2]} voxels &nbsp; | &nbsp; "
    f"spacing {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} "
    f"{case.image.geometry.spatial_unit} &nbsp; | &nbsp; "
    f"{len(daughters)} {'branch' if len(daughters) == 1 else 'branches'} found"
)
if detection_error:
    st.error(f"Detection failed: {detection_error}")

panel = st.segmented_control(
    "Panel",
    ["Simple View", "Detailed View", "VTK Snapshot", "Results", "Benchmark Results"],
    default="Simple View",
    label_visibility="collapsed",
    width="stretch",
    key="main_panel",
)

if panel != "VTK Snapshot":
    old_session = st.session_state.pop("vtk_session", None)
    if old_session is not None:
        old_session.close()
    st.session_state.pop("vtk_session_key", None)

if panel == "Simple View":
    with st.sidebar:
        with st.expander("03 · SIMPLE VIEW SETTINGS", expanded=False):
            frame = (
                st.slider("Frame", 0, shape[3] - 1, 0, key="fast_frame")
                if len(shape) == 4
                else 0
            )
            window = st.slider(
                "Window width",
                1,
                2000,
                600 if can_detect else 400,
                help="Width of the visible CT intensity range. Values are HU when the scan is calibrated.",
                key="fast_window",
            )
            level = st.slider(
                "Window center",
                -1000,
                1500,
                200 if can_detect else 40,
                help="Middle of the visible CT intensity range.",
                key="fast_level",
            )
            density = st.select_slider(
                "CT sampling limit",
                options=[48, 64, 72, 96],
                value=72,
                help="Higher values show more CT points but take longer to prepare.",
            )
            show_volume = st.checkbox("Show CT sample", True, key="fast_volume")
            show_mask = st.checkbox(
                "Show aorta surface", True, disabled=case.mask is None, key="fast_mask"
            )
    with st.container():
        with st.spinner("Preparing the scan view…"):
            payload = scene_payload(
                case,
                frame=frame,
                window=window,
                level=level,
                show_volume=show_volume,
                show_mask=show_mask,
                sampling_limit=density,
                prediction=prediction,
                show_branches=show_branches,
                selected_branch=selected_branch,
            )
        components.html(scene_html(payload), height=604, scrolling=False)
        st.caption(
            "The blue dot marks the ostium; the teal arrow points into the branch; the pale teal ring shows its estimated radius. "
            "This view samples the CT; use Detailed View to inspect the full volume."
        )
elif panel == "Detailed View":
    frame = (st.sidebar.slider("Frame", 0, shape[3] - 1, 0, key="detailed_frame")
             if len(shape) == 4 and shape[3] > 1 else 0)
    try:
        with st.spinner("Preparing the scan for interactive viewing…"):
            payload = load_browser_volume(image, mask, image_mtime, mask_mtime, frame)
        detailed_view(payload, prediction=prediction, selected_branch=selected_branch,
                      show_branches=show_branches)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        st.error(f"Interactive view could not load: {error}")
    st.caption("Use the controls inside the viewer for live navigation. VTK Snapshot retains "
               "the native volume renderer and its export controls.")
elif panel == "VTK Snapshot":
    with st.sidebar:
        with st.expander("03 · DETAILED VIEW SETTINGS", expanded=False):
            frame = (
                st.slider("Frame", 0, shape[3] - 1, 0, key="vtk_frame")
                if len(shape) == 4
                else 0
            )
            window = st.slider(
                "Window width",
                1,
                2000,
                600 if can_detect else 400,
                help="Width of the visible CT intensity range. Values are HU when the scan is calibrated.",
                key="vtk_window",
            )
            level = st.slider(
                "Window center",
                -1000,
                1500,
                200 if can_detect else 40,
                help="Middle of the visible CT intensity range.",
                key="vtk_level",
            )
            use_threshold = st.checkbox("Use minimum intensity", False)
            threshold = st.slider(
                "Minimum intensity", -1000, 1500, 100, disabled=not use_threshold
            )
            opacity = st.slider("Volume opacity", 0.0, 0.5, 0.12, 0.01)
            denoise = st.checkbox(
                "Denoise CT haze",
                False,
                help="Replace voxels without enough similarly bright immediate 3-D neighbors.",
            )
            denoise_tolerance = st.slider(
                "Denoise brightness tolerance", 0.0, 200.0, 40.0, 5.0, disabled=not denoise
            )
            denoise_min_neighbors = st.slider(
                "Denoise similar neighbors", 1, 26, 2, disabled=not denoise
            )
            max_dimension = st.select_slider(
                "CT sampling limit", [64, 96, 128, 192, 256], value=128
            )
            show_volume = st.checkbox("Show CT volume", False if can_detect else True)
            show_mask = st.checkbox(
                "Show aorta surface", True, disabled=case.mask is None
            )
            mip = st.checkbox("Maximum intensity projection", False)
        with st.expander("04 · VIEW ANGLE", expanded=False):
            azimuth = st.slider("Azimuth (degrees)", -180, 180, 30)
            elevation = st.slider("Elevation (degrees)", -90, 90, 25)
            zoom = st.slider("Zoom", 0.5, 2.5, 1.0, 0.1)
            focus_mask = st.checkbox(
                "Focus camera on aorta", bool(daughters), disabled=case.mask is None
            )
        with st.expander("05 · CROSS-SECTIONS", expanded=True):
            show_slices = st.checkbox("Show three orthogonal slice panels", True)
            show_planes = st.checkbox("Show slice planes in 3D", True)
            plane_opacity = st.slider(
                "3D plane opacity", 0.0, 1.0, 0.85, 0.05, disabled=not show_planes
            )
            st.caption(
                "I, J, and K move through the scan one voxel at a time. The letter in parentheses gives the physical axis direction."
            )
            slice_indices = tuple(
                st.slider(
                    f"{axis} / axis {dim} ({case.image.geometry.axis_codes[dim]})",
                    0,
                    shape[dim] - 1,
                    shape[dim] // 2,
                    disabled=not (show_slices or show_planes),
                )
                for dim, axis in enumerate("IJK")
            )
            cut = st.checkbox("Clip CT volume at K", False)
            cut_at_k = (
                st.slider(
                    "CT cut position / K",
                    0,
                    shape[2] - 1,
                    shape[2] // 2,
                    disabled=not cut,
                )
                if cut
                else None
            )

    with st.container():
        case_key = (image, mask, image_mtime, mask_mtime)
        view_key = (
            case_key,
            detector,
            selected_branch,
            show_branches,
            frame,
            window,
            level,
            threshold if use_threshold else None,
            denoise,
            denoise_tolerance,
            denoise_min_neighbors,
            opacity,
            max_dimension,
            azimuth,
            elevation,
            zoom,
            focus_mask,
            show_volume,
            show_mask,
            mip,
            cut_at_k,
            show_slices,
            show_planes,
            plane_opacity,
            slice_indices,
        )
        session_key = (
            case_key,
            frame,
            max_dimension,
            denoise,
            denoise_tolerance,
            denoise_min_neighbors,
        )
        if st.session_state.get("vtk_session_key") != session_key:
            old_session = st.session_state.pop("vtk_session", None)
            if old_session is not None:
                old_session.close()
            st.session_state["vtk_session_key"] = session_key
        if st.session_state.get("rendered_view") != view_key:
            try:
                options = VolumeViewOptions(
                    frame=frame,
                    max_dimension=max_dimension,
                    window=window,
                    level=level,
                    min_intensity=threshold if use_threshold else None,
                    denoise=denoise,
                    denoise_tolerance=denoise_tolerance,
                    denoise_min_neighbors=denoise_min_neighbors,
                    opacity=opacity,
                )
                started = monotonic()
                progress = st.progress(0, text="Rendering VTK volume · 0.0s")
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(
                        prepare_view,
                        case,
                        options,
                        azimuth=azimuth,
                        elevation=elevation,
                        zoom=zoom,
                        show_volume=show_volume,
                        show_mask=show_mask,
                        mip=mip,
                        cut_at_k=cut_at_k,
                        show_slices=show_slices,
                        show_planes=show_planes,
                        plane_opacity=plane_opacity,
                        focus_mask=focus_mask,
                        slice_indices=slice_indices,
                        branches=daughters,
                        show_branches=show_branches,
                        selected_branch=selected_branch,
                        session=st.session_state.get("vtk_session"),
                    )
                    while not pending.done():
                        elapsed = monotonic() - started
                        progress.progress(
                            min(95, 5 + int(elapsed * 3)),
                            text=f"Rendering VTK volume · {elapsed:.1f}s",
                        )
                        sleep(0.15)
                    png, mode, reason, counts, session = pending.result()
                elapsed = monotonic() - started
                progress.progress(100, text=f"3D view ready · {elapsed:.1f}s")
                st.session_state.update(
                    preview_png=png,
                    render_mode=mode,
                    fallback_reason=reason,
                    point_counts=counts,
                    render_seconds=elapsed,
                    rendered_view=view_key,
                    vtk_session=session,
                )
            except (OSError, ValueError, RuntimeError, ImportError) as error:
                st.session_state.pop("preview_png", None)
                st.error(f"3D view failed: {error}")
        if (
            st.session_state.get("rendered_view") == view_key
            and "preview_png" in st.session_state
        ):
            mode = st.session_state["render_mode"]
            st.caption(
                f"{mode} rendered in {st.session_state['render_seconds']:.1f}s. "
                "Move the I, J, or K sliders to inspect another cross-section."
            )
            if st.session_state["fallback_reason"]:
                st.info(
                    f"VTK unavailable: {st.session_state['fallback_reason']} CPU preview shown."
                )
            st.image(st.session_state["preview_png"], width="stretch")
            if show_branches and daughters:
                st.caption(
                    "Blue dot: ostium · Teal arrow: direction · Pale teal ring: estimated radius. "
                    "The markers show candidate measurements, not a segmented branch surface."
                )
            st.download_button(
                "Save PNG",
                st.session_state["preview_png"],
                file_name=f"{case.case_id}_3d.png",
                mime="image/png",
                on_click="ignore",
            )

if panel == "Results":
    st.subheader("Branch results")
    if prediction is not None:
        st.download_button(
            "Export evaluator JSON",
            json.dumps(prediction, indent=2) + "\n",
            file_name=f"{case.case_id}_{detector}_prediction.json",
            mime="application/json",
            on_click="ignore",
        )
    if detection_error:
        st.error(detection_error)
    elif not can_detect:
        st.info("Select a 3-D CT and its aorta mask to look for branches.")
    elif not daughters:
        st.info("No branches met this detector's criteria in the selected scan.")
    else:
        st.write(
            f"{len(daughters)} {'branch' if len(daughters) == 1 else 'branches'} found with the {detector} detector."
        )
        with st.popover("ⓘ Measurement guide"):
            st.write(
                "The ostium is where the branch meets the aorta. The seed is about 5 mm farther into the branch. "
                "Positions use physical coordinates in millimetres; direction is a unit vector pointing away from the aorta."
            )
        ordered = sorted(
            daughters, key=lambda item: item["instance_id"] != selected_branch
        )
        for daughter in ordered:
            fmt = lambda values: "  ".join(f"{value:+.2f}" for value in values)
            st.markdown(f"**{daughter['instance_id'].upper()}**")
            st.code(
                f"parent       {daughter['parent_instance_id']}\n"
                f"ostium       {fmt(daughter['ostium_xyz_mm'])} mm\n"
                f"seed         {fmt(daughter['seed_xyz_mm'])} mm\n"
                f"radius       {daughter['radius_mm']:.2f} mm\n"
                f"direction    {fmt(daughter['direction_xyz'])} (unit)",
                language=None,
            )
