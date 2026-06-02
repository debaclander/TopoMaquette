import base64
import io
import json
import math
import os
import tempfile
import zipfile

import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from shapely.affinity import scale as scale_geometry
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, snap, triangulate, unary_union

from core.diagnostics import build_dxf_diagnostics
from core.elevation_points import estimate_polygon_elevations_from_points
from core.exporter import (
    DEFAULT_LAYER_CONFIG,
    EXPORT_LINEWORK_SIMPLIFY_TOLERANCE_MM,
    diagnose_exported_dxf,
    export_sheets_to_combined_dxf,
    export_sheets_to_dxf,
    export_sheets_to_svg,
)
from core.nesting import perform_nesting
from core.parser import extract_3d_points_from_dxf, extract_polygons_from_dxf
from core.slicer import generate_rings
from core.terrain_slicer import (
    SOLID_VISIBILITY_RULE_VERSION,
    _annotate_solid_slice_visibility,
    _convert_solid_slices_to_hollow,
    build_contour_correction_table,
    generate_terrain_layers_from_contours,
)
from core.topology import build_topology_tree
from core.ui_utils import (
    contour_id_sort_key,
    normalize_override_map as _normalize_override_map,
    parse_bool as _parse_bool,
    parse_curve_ids as _parse_curve_ids,
    parse_float_or_none as _as_float_or_none,
    parse_int_or_none as _as_int_or_none,
    resolve_boundary_and_contours as _resolve_boundary_and_contours,
    resolve_effective_overrides as _resolve_effective_overrides,
)


ABOUT_THIS_SITE_TEXT = """
Ferramenta Streamlit para transformar ficheiros DXF em camadas de maquete,
preparar folhas de corte laser e exportar resultados em DXF/SVG.

Inclui modos solido e vazado, margem de base/cola, linhas de gravacao para
montagem, preview 2D/3D, nesting automatico e geracao opcional de base ate a
cota real.

Projeto em desenvolvimento ativo. Antes de cortar, confirme sempre os
ficheiros exportados num software CAD.

Licenca: GNU General Public License v3.0 ou posterior (GPL-3.0-or-later).
"""


if get_script_run_ctx() is None:
    print("Esta aplicacao deve ser executada com: streamlit run main.py")
    raise SystemExit(0)

st.set_page_config(
    page_title="Maquete Generator",
    layout="wide",
    menu_items={"About": ABOUT_THIS_SITE_TEXT},
)
st.title("Processador de DXF para Maquetes")

DEFAULT_PREVIEW_GRID_COLUMNS = 3
LAYER_PREVIEW_HEIGHT = 300
SHEET_PREVIEW_HEIGHT = 240
LAYER_PREVIEW_HEIGHT_BY_COLUMNS = {
    1: 760,
    2: 560,
    3: 400,
    4: 320,
}
SHEET_PREVIEW_HEIGHT_BY_COLUMNS = {
    1: 780,
    2: 560,
    3: 400,
    4: 320,
}
MODEL_3D_PREVIEW_HEIGHT = 760
LAYER_PREVIEW_CUT_LINE_WIDTH = 1.35
LAYER_PREVIEW_ROLE_LINE_WIDTH = 1.15
HOLLOW_SUPPORT_RULE_VERSION = "visible_buffer_clip_clean_restore_v1"
OFFSET_SIMPLIFY_RATIO = 0.02

session_defaults = {
    "polygons": None,
    "topology": None,
    "base_id": None,
    "direction_changes": set,
    "interval": 0.5,
    "target_layer_count": 37,
    "extend_base_to_zero": False,
    "all_topographic_points": list,
    "topographic_points": list,
    "selected_contour_layers": list,
    "selected_boundary_layer": None,
    "selected_points_layer": None,
    "slices": None,
    "active_layers_key": None,
    "manual_contour_overrides_points": dict,
    "manual_contour_overrides_summary": dict,
    "manual_contour_overrides": dict,
    "manual_override_last_source": "points",
    "dxf_diagnostics": None,
    "scale_calc_in_left": 1.0,
    "scale_calc_in_right": 1.0,
    "scale_calc_in_unit": "m",
    "scale_calc_out_left": 1.0,
    "scale_calc_out_right": 250.0,
    "scale_calc_out_unit": "m",
    "scale_calc_real_value": 0.5,
    "scale_calc_real_unit": "m",
    "scale_calc_model_value": 2.0,
    "scale_calc_model_unit": "mm",
    "scale_calc_file_measure": 0.5,
    "layer_preview_grid_columns": DEFAULT_PREVIEW_GRID_COLUMNS,
    "sheet_preview_grid_columns": DEFAULT_PREVIEW_GRID_COLUMNS,
}
for key, default in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default() if callable(default) else default


legacy_overrides = _normalize_override_map(st.session_state.get("manual_contour_overrides", {}))
points_overrides_state = _normalize_override_map(st.session_state.get("manual_contour_overrides_points", {}))
summary_overrides_state = _normalize_override_map(st.session_state.get("manual_contour_overrides_summary", {}))
if legacy_overrides and not points_overrides_state and not summary_overrides_state:
    points_overrides_state = dict(legacy_overrides)

st.session_state["manual_contour_overrides_points"] = points_overrides_state
st.session_state["manual_contour_overrides_summary"] = summary_overrides_state
st.session_state["manual_contour_overrides"] = _resolve_effective_overrides(
    points_overrides_state,
    summary_overrides_state,
    st.session_state.get("manual_override_last_source"),
)


def _generate_slices_from_points(
    polygons,
    selected_boundary_layer,
    topographic_points,
    interval,
    elevation_group_precision,
    scale_factor,
    slicing_mode,
    manual_overrides,
    target_layer_count,
    assembly_offset_mm,
    hollow_glue_margin_mm,
    point_curve_distance,
    extend_base_to_zero,
):
    boundary_geom, contour_polygons, _, _ = _resolve_boundary_and_contours(
        polygons,
        selected_boundary_layer,
    )
    if boundary_geom is None or not contour_polygons:
        return []

    engraving_offset = assembly_offset_mm / scale_factor if scale_factor > 0 else assembly_offset_mm
    glue_margin = hollow_glue_margin_mm / scale_factor if scale_factor > 0 else hollow_glue_margin_mm
    return generate_terrain_layers_from_contours(
        contour_polygons,
        topographic_points,
        boundary_geom,
        interval=interval,
        elevation_precision=elevation_group_precision,
        scale_factor=1.0,
        slicing_mode=slicing_mode,
        manual_overrides=manual_overrides,
        target_layer_count=int(target_layer_count),
        assembly_offset_mm=engraving_offset,
        hollow_glue_margin=glue_margin if slicing_mode == "hollow" else 0.0,
        near_distance=point_curve_distance,
        extend_base_to_zero=extend_base_to_zero,
    )


def _safe_repeat_count(item):
    try:
        return max(1, int(item.get("repeat_count", 1) or 1))
    except (TypeError, ValueError):
        return 1


def _scale_single_slice_for_output(item, scale_factor):
    row = dict(item)
    if scale_factor != 1.0:
        row["slice_geom"] = scale_geometry(
            item["slice_geom"],
            xfact=scale_factor,
            yfact=scale_factor,
            origin=(0, 0),
        )
        for geom_key in ("visible_geom", "support_geom", "hidden_base_geom"):
            geom_value = item.get(geom_key)
            if geom_value is not None:
                row[geom_key] = scale_geometry(
                    geom_value,
                    xfact=scale_factor,
                    yfact=scale_factor,
                    origin=(0, 0),
                )
        row["engraving_lines"] = [
            scale_geometry(line, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
            for line in item.get("engraving_lines", [])
        ]
        row["label_areas"] = [
            scale_geometry(area, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
            for area in item.get("label_areas", []) or []
        ]
    else:
        row["engraving_lines"] = list(item.get("engraving_lines", []))
        row["label_areas"] = list(item.get("label_areas", []) or [])
    return row


def _expand_repeated_slice(item):
    repeat_count = _safe_repeat_count(item)
    if not item.get("is_zero_base") or repeat_count <= 1:
        return [dict(item)]

    try:
        start_elevation = float(item.get("repeat_start_elevation", item.get("source_elevation", 0.0)) or 0.0)
    except (TypeError, ValueError):
        start_elevation = 0.0
    try:
        repeat_interval = float(item.get("repeat_interval", st.session_state.get("interval", 0.5)) or 0.5)
    except (TypeError, ValueError):
        repeat_interval = 0.5

    expanded = []
    for index in range(repeat_count):
        elevation = round(start_elevation + index * repeat_interval, 3)
        row = dict(item)
        row["id"] = f"{item.get('id', 'base')}_{index + 1}"
        row["elevation"] = elevation
        row["source_elevation"] = elevation
        row["repeat_count"] = 1
        row["repeat_index"] = index + 1
        row["repeat_total"] = repeat_count
        expanded.append(row)
    return expanded


def _scale_slices_for_output(slices, scale_factor, expand_repeats=False):
    prepared = []
    for item in slices:
        rows = _expand_repeated_slice(item) if expand_repeats else [dict(item)]
        for row in rows:
            prepared.append(_scale_single_slice_for_output(row, scale_factor))
    return prepared


def _normalize_hollow_support_rule(slices, glue_margin):
    if not slices:
        return slices, False

    glue_margin = float(glue_margin or 0.0)
    normalized = []
    changed = False
    for item in slices:
        row = dict(item)
        try:
            row_margin = float(row.get("support_glue_margin", 0.0) or 0.0)
        except (TypeError, ValueError):
            row_margin = 0.0
        row_integrity_ok = _hollow_row_integrity_ok(row)
        if (
            row.get("support_rule_version") == HOLLOW_SUPPORT_RULE_VERSION
            and abs(row_margin - glue_margin) <= 1e-9
            and row_integrity_ok
        ):
            normalized.append(row)
            continue

        visible_geom = row.get("visible_geom")
        hidden_base_geom = row.get("hidden_base_geom")
        support_geom = row.get("support_geom")
        base_candidate = hidden_base_geom if hidden_base_geom is not None and not hidden_base_geom.is_empty else support_geom
        if (
            (base_candidate is None or base_candidate.is_empty)
            and visible_geom is not None
            and not visible_geom.is_empty
        ):
            slice_geom = row.get("slice_geom")
            if slice_geom is not None and not slice_geom.is_empty:
                inferred_base = _polygonal_geometry(slice_geom.difference(visible_geom))
                if inferred_base is not None and not inferred_base.is_empty:
                    base_candidate = inferred_base
        if (
            visible_geom is None
            or visible_geom.is_empty
            or base_candidate is None
            or base_candidate.is_empty
        ):
            row["support_rule_version"] = HOLLOW_SUPPORT_RULE_VERSION
            row["support_glue_margin"] = glue_margin
            normalized.append(row)
            changed = True
            continue

        new_support = _contact_support_strip(visible_geom, base_candidate, glue_margin)
        row["support_geom"] = new_support
        row["slice_geom"] = visible_geom if new_support.is_empty else unary_union([visible_geom, new_support])
        row["hidden_base_geom"] = base_candidate
        row["support_rule_version"] = HOLLOW_SUPPORT_RULE_VERSION
        row["support_glue_margin"] = glue_margin
        normalized.append(row)
        changed = True

    return normalized, changed


def _hollow_row_integrity_ok(row):
    visible_geom = row.get("visible_geom")
    support_geom = row.get("support_geom")
    slice_geom = row.get("slice_geom")
    if visible_geom is None or visible_geom.is_empty or slice_geom is None or slice_geom.is_empty:
        return False

    support_geom = _polygonal_geometry(support_geom)
    rebuilt_slice = visible_geom if support_geom.is_empty else _polygonal_geometry(unary_union([visible_geom, support_geom]))
    if rebuilt_slice.is_empty:
        return False
    if _geometry_area_changed(slice_geom, rebuilt_slice):
        return False
    if not support_geom.is_empty:
        visible_touch_band = visible_geom.buffer(1e-6)
        for support_part in _polygon_parts(support_geom):
            if support_part.is_empty:
                continue
            # Every support component must touch the visible final shape.
            if not support_part.intersects(visible_touch_band):
                return False
    return True


def _prune_floating_hollow_rows(rows, glue_margin):
    if not rows or len(rows) < 2:
        return rows, False

    changed = False
    contact_tolerance = max(1e-6, min(max(float(glue_margin or 0.0) * 0.02, 0.01), 0.2))

    for index in range(1, len(rows)):
        lower_geom = _polygonal_geometry(rows[index - 1].get("slice_geom"))
        ring_geom = _polygonal_geometry(rows[index].get("slice_geom"))
        if lower_geom.is_empty or ring_geom.is_empty:
            continue

        visible_geom = _polygonal_geometry(rows[index].get("visible_geom"))
        if visible_geom.is_empty:
            continue

        lower_contact_band = lower_geom.buffer(
            contact_tolerance,
            cap_style=2,
            join_style=2,
            mitre_limit=2.0,
        )
        row = rows[index]
        hidden_base_geom = _polygonal_geometry(row.get("hidden_base_geom"))
        support_geom = _polygonal_geometry(row.get("support_geom"))
        base_candidate = hidden_base_geom if not hidden_base_geom.is_empty else support_geom
        if base_candidate.is_empty:
            inferred_base = _polygonal_geometry(ring_geom.difference(visible_geom))
            base_candidate = inferred_base if not inferred_base.is_empty else Polygon()

        pre_support = _contact_support_strip(visible_geom, base_candidate, glue_margin)
        pre_ring = visible_geom if pre_support.is_empty else _polygonal_geometry(unary_union([visible_geom, pre_support]))
        ring_parts = _polygon_parts(pre_ring)
        kept_parts = [
            part
            for part in ring_parts
            if part.intersects(lower_contact_band)
        ]
        if not kept_parts or len(kept_parts) == len(ring_parts):
            continue

        pruned_ring = _polygonal_geometry(unary_union(kept_parts))
        if pruned_ring.is_empty:
            continue

        visible_geom = _polygonal_geometry(visible_geom.intersection(pruned_ring))

        new_hidden_base = _polygonal_geometry(base_candidate.intersection(pruned_ring))
        new_support = _contact_support_strip(visible_geom, new_hidden_base, glue_margin)
        rebuilt_ring = visible_geom if new_support.is_empty else _polygonal_geometry(unary_union([visible_geom, new_support]))
        if rebuilt_ring.is_empty:
            continue

        row["slice_geom"] = rebuilt_ring
        row["visible_geom"] = visible_geom
        row["support_geom"] = new_support
        row["hidden_base_geom"] = new_hidden_base
        rows[index] = row
        changed = True

    return rows, changed


def _ensure_hollow_slice_geometry(slices, glue_margin, boundary_geom):
    if not slices:
        return slices, False

    has_current_hollow_rule = all(
        item.get("support_rule_version") == HOLLOW_SUPPORT_RULE_VERSION
        for item in slices
    )
    if has_current_hollow_rule:
        return slices, False

    has_hollow_base_metadata = any(
        (item.get("hidden_base_geom") is not None)
        and (not item["hidden_base_geom"].is_empty)
        for item in slices
    )
    if has_hollow_base_metadata:
        return slices, False

    converted = _convert_solid_slices_to_hollow(
        slices,
        area_tolerance=0.1,
        glue_margin=float(glue_margin or 0.0),
        glue_limit_geom=boundary_geom,
    )
    return converted, bool(converted)


def _ensure_solid_slice_visibility(slices):
    if not slices:
        return slices, False, False

    has_current_solid_visibility = all(
        item.get("support_rule_version") == SOLID_VISIBILITY_RULE_VERSION
        for item in slices
    )
    if has_current_solid_visibility:
        return slices, False, False

    converted = _annotate_solid_slice_visibility(slices, area_tolerance=0.1)
    geometry_changed = False
    for before, after in zip(slices, converted):
        before_geom = before.get("slice_geom")
        after_geom = after.get("slice_geom")
        if _geometry_area_changed(before_geom, after_geom):
            geometry_changed = True
            break

    return converted, bool(converted), geometry_changed


def _geometry_area_changed(before_geom, after_geom):
    if before_geom is None or before_geom.is_empty:
        return after_geom is not None and not after_geom.is_empty
    if after_geom is None or after_geom.is_empty:
        return True
    try:
        return before_geom.symmetric_difference(after_geom).area > 1e-6
    except Exception:
        return abs(before_geom.area - after_geom.area) > 1e-6


def _contact_support_strip(visible_geom, base_candidate_geom, glue_margin):
    if (
        visible_geom is None
        or visible_geom.is_empty
        or base_candidate_geom is None
        or base_candidate_geom.is_empty
        or glue_margin is None
        or glue_margin <= 0
    ):
        return Polygon()

    support = visible_geom.buffer(
        float(glue_margin),
        join_style=2,
        mitre_limit=2.0,
    ).intersection(base_candidate_geom)

    support = _clean_support_geometry(support, visible_geom, glue_margin)
    if support.is_empty:
        return Polygon()
    return support


def _shared_visible_base_contact_line(visible_geom, base_candidate_geom, glue_margin):
    if (
        visible_geom is None
        or visible_geom.is_empty
        or base_candidate_geom is None
        or base_candidate_geom.is_empty
    ):
        return LineString()

    shared = visible_geom.boundary.intersection(base_candidate_geom.boundary)
    if shared.is_empty:
        shared = visible_geom.boundary.intersection(base_candidate_geom)
    if shared.is_empty:
        return LineString()

    return _simplify_contact_line(shared, glue_margin)


def _clean_support_geometry(support_geom, visible_geom, glue_margin):
    support_geom = _polygonal_geometry(support_geom)
    if support_geom.is_empty:
        return Polygon()

    glue_margin = float(glue_margin or 0.0)
    contact_tolerance = max(1e-6, min(glue_margin * 0.02, 0.1))
    min_contact_length = max(0.25, glue_margin * 0.05)
    min_width = max(0.2, glue_margin * 0.15)
    min_area = max(1e-6, glue_margin * glue_margin * 0.05)
    visible_contact_band = visible_geom.boundary.buffer(
        contact_tolerance,
        cap_style=2,
        join_style=2,
        mitre_limit=2.0,
    )

    kept_parts = []
    for part in _polygon_parts(support_geom):
        if part.area <= min_area:
            continue
        if _support_part_contact_length(part, visible_contact_band) < min_contact_length:
            continue
        if _support_part_effective_width(part) < min_width:
            continue
        kept_parts.append(part)

    if not kept_parts:
        return Polygon()

    cleaned = unary_union(kept_parts)
    if not cleaned.is_valid:
        cleaned = cleaned.buffer(0)
    return _polygonal_geometry(cleaned)


def _support_part_contact_length(part, visible_contact_band):
    if part is None or part.is_empty or visible_contact_band is None or visible_contact_band.is_empty:
        return 0.0
    contact = part.boundary.intersection(visible_contact_band)
    return sum(line.length for line in _linear_parts_2d(contact))


def _support_part_effective_width(part):
    if part is None or part.is_empty or part.area <= 0:
        return 0.0

    widths = []
    if part.length > 1e-9:
        widths.append((2.0 * part.area) / part.length)

    rect = part.minimum_rotated_rectangle
    if not rect.is_empty and rect.geom_type == "Polygon":
        coords = list(rect.exterior.coords)
        edge_lengths = [
            math.hypot(coords[idx + 1][0] - coords[idx][0], coords[idx + 1][1] - coords[idx][1])
            for idx in range(min(4, len(coords) - 1))
        ]
        positive_lengths = [length for length in edge_lengths if length > 1e-9]
        if positive_lengths:
            widths.append(min(positive_lengths))

    return min(widths) if widths else 0.0


def _offset_contact_line_to_base_side(contact_line, base_candidate_geom, glue_margin):
    if _contact_margin_covers_base(contact_line, base_candidate_geom, glue_margin):
        return base_candidate_geom

    supports = []
    for line in _linear_parts_2d(contact_line):
        support = _best_base_side_single_offset(line, base_candidate_geom, glue_margin)
        if support is not None and not support.is_empty:
            supports.append(support)

    if not supports:
        return Polygon()

    support = unary_union(supports).intersection(base_candidate_geom)
    return _polygonal_geometry(support)


def _contact_margin_covers_base(contact_line, base_candidate_geom, glue_margin):
    if (
        contact_line is None
        or contact_line.is_empty
        or base_candidate_geom is None
        or base_candidate_geom.is_empty
    ):
        return False
    coverage = contact_line.buffer(
        float(glue_margin),
        cap_style=2,
        join_style=2,
        mitre_limit=2.0,
    )
    if coverage.is_empty:
        return False
    remainder = base_candidate_geom.difference(coverage)
    return remainder.is_empty or remainder.area <= max(1e-6, base_candidate_geom.area * 1e-6)


def _best_base_side_single_offset(line, base_candidate_geom, glue_margin):
    line = _normalized_line_for_offset(line)
    if line is None or line.is_empty or line.length <= 1e-9:
        return None

    best_support = None
    best_score = None
    expected_area = max(line.length * float(glue_margin), 1e-9)
    for distance in (float(glue_margin), -float(glue_margin)):
        try:
            strip = line.buffer(
                distance,
                single_sided=True,
                join_style=2,
                mitre_limit=2.0,
            )
        except (AttributeError, ValueError, TypeError):
            continue

        if strip.is_empty or strip.area <= 1e-9:
            continue
        support = strip.intersection(base_candidate_geom)
        area = support.area if not support.is_empty else 0.0
        if area <= 1e-9:
            continue
        inside_ratio = area / max(strip.area, 1e-9)
        area_error = abs(area - expected_area) / expected_area
        score = (inside_ratio, -area_error, area)
        if best_score is None or score > best_score:
            best_score = score
            best_support = support

    return best_support


def _normalized_line_for_offset(line, tolerance=1e-9):
    if line is None or line.is_empty:
        return None
    try:
        coords = list(line.coords)
    except Exception:
        return None
    if len(coords) < 2:
        return None

    cleaned = [coords[0]]
    for point in coords[1:]:
        if (
            abs(point[0] - cleaned[-1][0]) <= tolerance
            and abs(point[1] - cleaned[-1][1]) <= tolerance
        ):
            continue
        cleaned.append(point)

    if len(cleaned) < 2:
        return None
    if (
        len(cleaned) == 2
        and abs(cleaned[0][0] - cleaned[1][0]) <= tolerance
        and abs(cleaned[0][1] - cleaned[1][1]) <= tolerance
    ):
        return None

    try:
        normalized = LineString(cleaned)
    except (TypeError, ValueError):
        return None

    if normalized.is_empty or normalized.length <= tolerance:
        return None
    return normalized


def _line_inside_ratio(line, geom):
    if line is None or line.is_empty or geom is None or geom.is_empty:
        return 0.0
    total_length = line.length
    if total_length <= 1e-9:
        return 0.0
    inside = line.intersection(geom)
    return max(0.0, min(1.0, inside.length / total_length))


def _strip_polygons_between_lines(line, offset_line):
    if line is None or offset_line is None or line.is_empty or offset_line.is_empty:
        return []

    line_coords = list(line.coords)
    offset_coords = list(offset_line.coords)
    if len(line_coords) < 2 or len(offset_coords) < 2:
        return []

    if _is_closed_line(line) and _is_closed_line(offset_line):
        original_poly = _safe_polygon(line_coords)
        offset_poly = _safe_polygon(offset_coords)
        if original_poly.is_empty or offset_poly.is_empty:
            return []
        return _polygon_parts(original_poly.symmetric_difference(offset_poly))

    candidates = []
    for candidate_offset_coords in (offset_coords, list(reversed(offset_coords))):
        shell = line_coords + list(reversed(candidate_offset_coords))
        if len(shell) < 4:
            continue
        poly = _safe_polygon(shell)
        candidates.extend(_polygon_parts(poly))
    return candidates


def _safe_polygon(coords):
    try:
        poly = Polygon(coords)
    except (TypeError, ValueError):
        return Polygon()
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if not poly.is_empty else Polygon()


def _is_closed_line(line):
    coords = list(line.coords)
    if len(coords) < 4:
        return False
    return bool(getattr(line, "is_ring", False)) or (
        abs(coords[0][0] - coords[-1][0]) <= 1e-9
        and abs(coords[0][1] - coords[-1][1]) <= 1e-9
    )


def _simplify_contact_line(geom, glue_margin):
    if geom is None or geom.is_empty or glue_margin is None or glue_margin <= 0:
        return LineString()

    tolerance = min(float(glue_margin) * OFFSET_SIMPLIFY_RATIO, 0.25)
    min_length = max(tolerance, 1e-6)
    source_lines = [
        line
        for line in _linear_parts_2d(geom)
        if line.length > min_length
    ]
    if not source_lines:
        return LineString()

    merged = unary_union(source_lines)
    merged = snap(merged, merged, max(tolerance, 1e-6))
    merged = unary_union(_linear_parts_2d(merged))
    try:
        merged = linemerge(merged)
    except ValueError:
        pass

    simplified_lines = []
    for line in _linear_parts_2d(merged):
        if line.length <= min_length:
            continue
        simplified = line.simplify(tolerance, preserve_topology=False) if tolerance > 0 else line
        if simplified.is_empty:
            simplified = line
        simplified_lines.extend(
            part for part in _linear_parts_2d(simplified) if part.length > min_length
        )

    if not simplified_lines:
        return LineString()

    merged = unary_union(simplified_lines)
    merged = snap(merged, merged, max(tolerance, 1e-6))
    merged = unary_union(_linear_parts_2d(merged))
    try:
        merged = linemerge(merged)
    except ValueError:
        pass
    return merged


def _linear_parts_2d(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type in ("LineString", "LinearRing"):
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts = []
        for child in geom.geoms:
            parts.extend(_linear_parts_2d(child))
        return parts
    return []


def _polygonal_geometry(geom):
    if geom is None or geom.is_empty:
        return Polygon()
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        parts = [
            part
            for child in geom.geoms
            for part in _polygon_parts(child)
        ]
        return unary_union(parts) if parts else Polygon()
    return Polygon()


def _polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return [
            part
            for child in geom.geoms
            for part in _polygon_parts(child)
        ]
    return []


def _extract_linework_2d(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type in ("LineString", "LinearRing"):
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts = []
        for child in geom.geoms:
            parts.extend(_extract_linework_2d(child))
        return parts
    if geom.geom_type == "Polygon":
        return [geom.exterior]
    if geom.geom_type == "MultiPolygon":
        return [poly.exterior for poly in geom.geoms]
    return []


def _clean_engraving_lines_2d(lines, minimum_length=2.0):
    if not lines:
        return []
    merged = unary_union(lines)
    if merged.geom_type == "MultiLineString":
        merged = linemerge(merged)
    return [
        line
        for line in _extract_linework_2d(merged)
        if line.length >= minimum_length
    ]


def _refresh_assembly_engraving_lines(rows, offset, slicing_mode=None):
    if not rows:
        return rows, False

    try:
        offset_value = float(offset or 0.0)
    except (TypeError, ValueError):
        offset_value = 0.0

    def _line_signature(line):
        minx, miny, maxx, maxy = line.bounds
        return (
            round(float(line.length), 6),
            round(float(minx), 4),
            round(float(miny), 4),
            round(float(maxx), 4),
            round(float(maxy), 4),
        )

    def _engraving_signature(lines):
        return tuple(sorted(_line_signature(line) for line in lines if line is not None and not line.is_empty))

    rebuilt = [dict(row) for row in rows]
    original_signatures = [
        _engraving_signature(item.get("engraving_lines", []) or [])
        for item in rebuilt
    ]
    for item in rebuilt:
        item["engraving_lines"] = []

    def _reference_geom(item):
        visible = item.get("visible_geom")
        if visible is not None and not visible.is_empty:
            return visible
        return item.get("slice_geom")

    def _target_geom(item):
        support = item.get("support_geom")
        if support is not None and not support.is_empty:
            return support
        return item.get("slice_geom")

    def _is_solid_row(item):
        return item.get("support_rule_version") == SOLID_VISIBILITY_RULE_VERSION

    def _is_hollow_row(item):
        return item.get("support_rule_version") == HOLLOW_SUPPORT_RULE_VERSION

    def _hollow_contact_lines(item):
        visible = item.get("visible_geom")
        support = item.get("support_geom")
        if (
            visible is None
            or visible.is_empty
            or support is None
            or support.is_empty
        ):
            return []

        contact = visible.boundary.intersection(support.boundary)
        if contact.is_empty:
            contact = visible.boundary.intersection(support)
        return _clean_engraving_lines_2d(_extract_linework_2d(contact))

    use_hollow_contact_rule = (
        slicing_mode == "hollow"
        or (slicing_mode is None and any(_is_hollow_row(item) for item in rebuilt))
    )
    if use_hollow_contact_rule:
        for item in rebuilt:
            lines = _hollow_contact_lines(item)
            if lines:
                item["engraving_lines"] = lines
        refreshed_signatures = [
            _engraving_signature(item.get("engraving_lines", []) or [])
            for item in rebuilt
        ]
        changed = refreshed_signatures != original_signatures
        return rebuilt, changed

    for index in range(len(rebuilt) - 1):
        current_row = rebuilt[index]
        upper_row = rebuilt[index + 1]

        use_solid_rule = (
            slicing_mode == "solid"
            or (
                slicing_mode is None
                and (_is_solid_row(current_row) or _is_solid_row(upper_row))
            )
        )

        # Keep Option A engraving tied to physical cut geometry.
        # Option B is selected explicitly and keeps support/visible guides.
        if use_solid_rule:
            current_geom = current_row.get("slice_geom")
            upper_ref_geom = upper_row.get("slice_geom")
        else:
            current_geom = _target_geom(current_row)
            upper_ref_geom = _reference_geom(upper_row)

        upper_physical_geom = upper_row.get("slice_geom")
        if (
            current_geom is None
            or current_geom.is_empty
            or upper_ref_geom is None
            or upper_ref_geom.is_empty
            or upper_physical_geom is None
            or upper_physical_geom.is_empty
        ):
            continue

        guide_source = (
            upper_ref_geom.buffer(-offset_value, join_style=2)
            if offset_value > 0
            else upper_ref_geom
        )
        if guide_source.is_empty:
            continue

        guide = guide_source.boundary.intersection(current_geom).intersection(upper_physical_geom)
        lines = _clean_engraving_lines_2d(_extract_linework_2d(guide))
        if lines:
            rebuilt[index]["engraving_lines"] = lines

    refreshed_signatures = [
        _engraving_signature(item.get("engraving_lines", []) or [])
        for item in rebuilt
    ]
    changed = refreshed_signatures != original_signatures
    return rebuilt, changed


def _scale_calc_number(key, fallback):
    try:
        return float(st.session_state.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)


_SCALE_CALC_UNITS_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
}


def _scale_calc_unit_factor(key, fallback="m"):
    unit = st.session_state.get(key, fallback)
    return _SCALE_CALC_UNITS_TO_MM.get(unit, _SCALE_CALC_UNITS_TO_MM[fallback])


def _scale_calc_real_mm():
    return max(_scale_calc_number("scale_calc_real_value", 0.5), 0.0) * _scale_calc_unit_factor("scale_calc_real_unit", "m")


def _scale_calc_model_mm():
    return max(_scale_calc_number("scale_calc_model_value", 2.0), 0.0) * _scale_calc_unit_factor("scale_calc_model_unit", "mm")


def _scale_calc_out_ratio():
    out_left = max(_scale_calc_number("scale_calc_out_left", 1.0), 1e-9)
    out_right = max(_scale_calc_number("scale_calc_out_right", 250.0), 1e-9)
    return out_right / out_left


def _preview_height_for_columns(columns, mapping, fallback):
    try:
        column_count = int(columns)
    except (TypeError, ValueError):
        column_count = DEFAULT_PREVIEW_GRID_COLUMNS
    column_count = max(1, min(4, column_count))
    return int(mapping.get(column_count, fallback))


def _preview_grid_columns_state(key, fallback=DEFAULT_PREVIEW_GRID_COLUMNS):
    try:
        column_count = int(st.session_state.get(key, fallback))
    except (TypeError, ValueError):
        column_count = fallback
    column_count = max(1, min(4, column_count))
    st.session_state[key] = column_count
    return column_count


def _scale_calc_in_mm_per_file_unit():
    in_left = max(_scale_calc_number("scale_calc_in_left", 1.0), 1e-9)
    in_right = max(_scale_calc_number("scale_calc_in_right", 1.0), 1e-9)
    return (in_right / in_left) * _scale_calc_unit_factor("scale_calc_in_unit", "m")


def _scale_calc_set_real_from_mm(real_mm):
    factor = _scale_calc_unit_factor("scale_calc_real_unit", "m")
    st.session_state["scale_calc_real_value"] = round(max(real_mm, 0.0) / factor, 6)


def _scale_calc_set_model_from_mm(model_mm):
    factor = _scale_calc_unit_factor("scale_calc_model_unit", "mm")
    st.session_state["scale_calc_model_value"] = round(max(model_mm, 0.0) / factor, 6)


def _scale_calc_update_file_measure_from_real():
    in_mm = max(_scale_calc_in_mm_per_file_unit(), 1e-9)
    st.session_state["scale_calc_file_measure"] = round(_scale_calc_real_mm() / in_mm, 6)


def _scale_calc_update_model_from_real_or_out():
    _scale_calc_update_file_measure_from_real()
    model_mm = _scale_calc_real_mm() / max(_scale_calc_out_ratio(), 1e-9)
    _scale_calc_set_model_from_mm(model_mm)


def _scale_calc_update_real_from_in():
    file_measure = max(_scale_calc_number("scale_calc_file_measure", 0.5), 0.0)
    real_mm = file_measure * _scale_calc_in_mm_per_file_unit()
    _scale_calc_set_real_from_mm(real_mm)
    _scale_calc_set_model_from_mm(real_mm / max(_scale_calc_out_ratio(), 1e-9))


def _scale_calc_update_out_from_model():
    _scale_calc_update_file_measure_from_real()
    real_mm = _scale_calc_real_mm()
    model_mm = max(_scale_calc_model_mm(), 1e-9)
    out_left = max(_scale_calc_number("scale_calc_out_left", 1.0), 1e-9)
    st.session_state["scale_calc_out_right"] = round(max(out_left * real_mm / model_mm, 0.0001), 6)


with st.sidebar.expander("About this site", expanded=False):
    st.markdown(ABOUT_THIS_SITE_TEXT)

st.sidebar.header("1. Upload de Ficheiro")
uploaded_file = st.sidebar.file_uploader("Ficheiro DXF", type=["dxf"])

if uploaded_file is not None and st.session_state["polygons"] is None:
    with st.spinner("A ler o ficheiro DXF e processar geometrias..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        polygons = extract_polygons_from_dxf(tmp_path)
        all_topographic_points = extract_3d_points_from_dxf(tmp_path, layer_name=None)
        dxf_diagnostics = build_dxf_diagnostics(tmp_path)

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        st.session_state["polygons"] = polygons
        st.session_state["all_topographic_points"] = all_topographic_points
        st.session_state["topographic_points"] = []
        st.session_state["manual_contour_overrides_points"] = {}
        st.session_state["manual_contour_overrides_summary"] = {}
        st.session_state["manual_contour_overrides"] = {}
        st.session_state["manual_override_last_source"] = "points"
        st.session_state["dxf_diagnostics"] = dxf_diagnostics

        if polygons:
            st.session_state["topology"] = build_topology_tree(polygons)
            st.success(f"DXF lido com sucesso! {len(polygons)} curvas encontradas.")
            if all_topographic_points:
                point_layers = sorted({p["layer"] for p in all_topographic_points})
                st.info(
                    f"Foram encontrados {len(all_topographic_points)} pontos 3D "
                    f"em {len(point_layers)} layer(s). Configure a layer de pontos no passo 1.5."
                )
        else:
            st.error("Nao foram encontradas polilinhas fechadas validas no DXF.")

diagnostics = st.session_state.get("dxf_diagnostics")
if diagnostics is not None:
    with st.expander("Diagnostico do DXF (closed, sobreposicoes, geometrias invalidas)", expanded=False):
        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Closed (bruto)", diagnostics.get("closed_total", 0))
        col_d2.metric("Closed validos", diagnostics.get("valid_total", 0))
        col_d3.metric("Closed invalidos", diagnostics.get("invalid_total", 0))

        overlap_summary = diagnostics.get("overlap_summary", {})
        col_o1, col_o2, col_o3 = st.columns(3)
        col_o1.metric("Sobreposicoes", overlap_summary.get("overlap", 0))
        col_o2.metric("Duplicados", overlap_summary.get("duplicate", 0))
        col_o3.metric("Nested", overlap_summary.get("nested", 0))

        layer_rows = diagnostics.get("layers", [])
        if layer_rows:
            st.caption("Resumo por layer")
            st.dataframe(layer_rows, hide_index=True, use_container_width=True)

        invalid_rows = diagnostics.get("invalid_shapes", [])
        if invalid_rows:
            st.caption("Geometrias invalidas / self-intersection")
            st.dataframe(
                [
                    {
                        "shape_id": row.get("shape_id"),
                        "layer": row.get("layer"),
                        "reason": row.get("reason"),
                        "closed_type": row.get("closed_type"),
                        "x": row.get("x"),
                        "y": row.get("y"),
                        "point_count": row.get("point_count"),
                    }
                    for row in invalid_rows
                ],
                hide_index=True,
                use_container_width=True,
            )

        overlap_rows = diagnostics.get("overlap_pairs", [])
        if overlap_rows:
            st.caption("Formas sobrepostas/duplicadas (amostra)")
            st.dataframe(overlap_rows, hide_index=True, use_container_width=True)

        polygons_for_diag = st.session_state.get("polygons") or []
        if polygons_for_diag or invalid_rows or overlap_rows:
            fig_diag = go.Figure()

            for polygon_data in polygons_for_diag:
                geometry = polygon_data["geometry"]
                x_vals, y_vals = geometry.exterior.xy
                fig_diag.add_trace(
                    go.Scatter(
                        x=list(x_vals),
                        y=list(y_vals),
                        mode="lines",
                        line=dict(color="#d0d0d0", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            for row in invalid_rows[:150]:
                points = row.get("points") or []
                if not points:
                    continue
                x_vals = [point[0] for point in points]
                y_vals = [point[1] for point in points]
                fig_diag.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines",
                        line=dict(color="#d62728", width=2),
                        name=f"Invalida {row.get('shape_id')}",
                        showlegend=False,
                        hovertemplate=(
                            f"Shape {row.get('shape_id')}<br>"
                            f"Layer: {row.get('layer')}<br>"
                            f"Erro: {row.get('reason')}<extra></extra>"
                        ),
                    )
                )

            if overlap_rows:
                fig_diag.add_trace(
                    go.Scatter(
                        x=[row.get("x") for row in overlap_rows],
                        y=[row.get("y") for row in overlap_rows],
                        mode="markers",
                        marker=dict(color="#ff7f0e", size=8, symbol="x"),
                        name="Sobreposicoes",
                        hovertemplate=(
                            "A:%{customdata[0]} (%{customdata[1]})<br>"
                            "B:%{customdata[2]} (%{customdata[3]})<br>"
                            "Tipo:%{customdata[4]}<br>"
                            "Area:%{customdata[5]}<extra></extra>"
                        ),
                        customdata=[
                            [
                                row.get("shape_a"),
                                row.get("layer_a"),
                                row.get("shape_b"),
                                row.get("layer_b"),
                                row.get("relation"),
                                row.get("intersection_area"),
                            ]
                            for row in overlap_rows
                        ],
                    )
                )

            if invalid_rows:
                fig_diag.add_trace(
                    go.Scatter(
                        x=[row.get("x") for row in invalid_rows],
                        y=[row.get("y") for row in invalid_rows],
                        mode="markers",
                        marker=dict(color="#d62728", size=9, symbol="circle-open"),
                        name="Invalidas",
                        hovertemplate=(
                            "Shape:%{customdata[0]}<br>"
                            "Layer:%{customdata[1]}<br>"
                            "Erro:%{customdata[2]}<extra></extra>"
                        ),
                        customdata=[
                            [row.get("shape_id"), row.get("layer"), row.get("reason")]
                            for row in invalid_rows
                        ],
                    )
                )

            fig_diag.update_layout(
                title="Mapa de diagnostico do DXF",
                xaxis=dict(scaleanchor="y", scaleratio=1),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_diag, use_container_width=True, key="dxf_diagnostics_map")

        diagnostics_export = {
            "closed_total": diagnostics.get("closed_total", 0),
            "valid_total": diagnostics.get("valid_total", 0),
            "invalid_total": diagnostics.get("invalid_total", 0),
            "layers": layer_rows,
            "invalid_shapes": [
                {key: value for key, value in row.items() if key != "points"}
                for row in invalid_rows
            ],
            "overlap_pairs": overlap_rows,
            "overlap_summary": overlap_summary,
        }
        st.download_button(
            label="Baixar diagnostico (JSON)",
            data=json.dumps(diagnostics_export, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="dxf_diagnostico.json",
            mime="application/json",
        )

if st.session_state["polygons"] is not None:
    all_layers = sorted({p["layer"] for p in st.session_state["polygons"]})
    all_topographic_points = st.session_state.get("all_topographic_points", [])
    point_layers = sorted({p["layer"] for p in all_topographic_points})
    st.sidebar.header("1.5. Filtrar Camadas")

    auto_boundary_label = "(Automatico: maior poligono)"
    boundary_options = [auto_boundary_label] + all_layers
    boundary_name_hints = {"boundary", "limite", "perimetro", "contorno"}
    default_boundary_layer = next((l for l in all_layers if l.lower() in boundary_name_hints), None)
    default_boundary_index = boundary_options.index(default_boundary_layer) if default_boundary_layer else 0
    previous_boundary_layer = st.session_state.get("selected_boundary_layer")
    if previous_boundary_layer in all_layers:
        default_boundary_index = boundary_options.index(previous_boundary_layer)

    selected_boundary_label = st.sidebar.selectbox(
        "Camada boundary:",
        options=boundary_options,
        index=default_boundary_index,
        help="Escolha a layer que representa o limite exterior da maquete.",
    )
    selected_boundary_layer = None if selected_boundary_label == auto_boundary_label else selected_boundary_label
    st.session_state["selected_boundary_layer"] = selected_boundary_layer

    contour_layer_options = [l for l in all_layers if l != selected_boundary_layer]
    topography_layers = [l for l in contour_layer_options if l.lower() in ("topografia", "topography", "topo")]
    ignored_default_layers = {"boundary", "camadas", "3d_pontos", "0"}
    if selected_boundary_layer:
        ignored_default_layers.add(selected_boundary_layer.lower())

    previous_contour_layers = st.session_state.get("selected_contour_layers", [])
    default_selected = [l for l in previous_contour_layers if l in contour_layer_options]
    if not default_selected:
        if topography_layers:
            default_selected = topography_layers
        else:
            default_selected = [l for l in contour_layer_options if l.lower() not in ignored_default_layers] or contour_layer_options

    selected_layers = st.sidebar.multiselect(
        "Camadas de curvas de nivel:",
        options=contour_layer_options,
        default=default_selected,
    )
    st.session_state["selected_contour_layers"] = selected_layers

    no_points_label = "(Nenhuma)"
    point_options = [no_points_label] + point_layers
    point_name_hints = {"3d_pontos", "3dpoints", "pontos_3d", "topografia_pontos"}
    default_point_layer = next((l for l in point_layers if l.lower() in point_name_hints), None)
    if default_point_layer is None and len(point_layers) == 1:
        default_point_layer = point_layers[0]

    default_point_index = point_options.index(default_point_layer) if default_point_layer else 0
    previous_points_layer = st.session_state.get("selected_points_layer")
    if previous_points_layer in point_layers:
        default_point_index = point_options.index(previous_points_layer)

    selected_points_label = st.sidebar.selectbox(
        "Camada de pontos 3D:",
        options=point_options,
        index=default_point_index,
        help="Selecione a layer dos POINT 3D para atribuir cotas automaticamente.",
    )
    selected_points_layer = None if selected_points_label == no_points_label else selected_points_label
    st.session_state["selected_points_layer"] = selected_points_layer

    if selected_points_layer is None:
        st.session_state["topographic_points"] = []
    else:
        st.session_state["topographic_points"] = [
            p for p in all_topographic_points if p["layer"] == selected_points_layer
        ]

    active_polygons = [
        p for p in st.session_state["polygons"]
        if (
            p["layer"] in selected_layers
            or (selected_boundary_layer is not None and p["layer"] == selected_boundary_layer)
            or (selected_boundary_layer is None and p["layer"].lower() == "boundary")
        )
    ]

    active_layers_key = (
        tuple(sorted(selected_layers)),
        selected_boundary_layer,
        selected_points_layer,
    )
    if active_polygons:
        if st.session_state.get("active_layers_key") != active_layers_key:
            st.session_state["topology"] = build_topology_tree(active_polygons)
            st.session_state["active_layers_key"] = active_layers_key
            st.session_state["slices"] = None
    else:
        st.session_state["topology"] = None
        st.session_state["active_layers_key"] = active_layers_key
        st.warning("Nao ha geometrias ativas. Selecione ao menos uma camada de curvas ou a boundary.")

if st.session_state["polygons"] is not None and st.session_state["topology"] is not None:
    st.sidebar.header("2. Definicoes")
    st.sidebar.subheader("2.1 Camadas")
    interval = st.sidebar.number_input(
        "Intervalo vertical (m)",
        value=float(st.session_state.get("interval", 0.5) or 0.5),
        min_value=0.0,
        step=0.5,
        help="Passo regular das curvas de nivel. Use 0 para inferir pelo numero de camadas.",
    )
    st.session_state["interval"] = interval
    target_layer_count = st.sidebar.number_input(
        "Numero total de camadas",
        value=int(st.session_state.get("target_layer_count", 37) or 37),
        min_value=1,
        step=1,
        help="Usado para validar ou inferir o intervalo quando o passo esta a 0.",
    )
    st.session_state["target_layer_count"] = int(target_layer_count)
    extend_base_to_zero = st.sidebar.checkbox(
        "Imprimir base ate a cota 0 real",
        value=bool(st.session_state.get("extend_base_to_zero", False)),
        help=(
            "Se a cota mais baixa for 70 e o intervalo for 0.5 m, acrescenta "
            "placas de base nas cotas 0, 0.5, 1.0... ate chegar a 70."
        ),
    )
    st.session_state["extend_base_to_zero"] = extend_base_to_zero
    elevation_group_precision = st.sidebar.number_input(
        "Forcar intervalo das cotas (m, 0 = usar intervalo vertical)",
        value=0.0,
        min_value=0.0,
        step=0.5,
    )
    point_curve_distance = st.sidebar.number_input(
        "Raio ponto-curva para cota direta (m, 0 = desligado)",
        value=float(st.session_state.get("point_curve_distance", 0.2) or 0.2),
        min_value=0.0,
        step=0.01,
        format="%.2f",
    )
    st.session_state["point_curve_distance"] = point_curve_distance
    dxf_units = st.sidebar.selectbox(
        "Unidades do DXF",
        ["Metros (m)", "Centimetros (cm)", "Milimetros (mm)"],
    )
    scale_denominator = st.sidebar.number_input("Escala Desejada (1:X)", value=500, step=50)
    with st.sidebar.expander("Calculadora de escalas", expanded=True):
        st.caption("Ferramenta de apoio; nao altera a escala usada pelo programa.")

        st.caption("IN ficheiro importado DXF")
        in_left_col, in_sep_col, in_right_col, in_unit_col = st.columns([1.0, 0.15, 1.0, 0.75])
        in_left_col.number_input(
            "IN valor 1",
            min_value=0.0001,
            step=1.0,
            format="%.4f",
            key="scale_calc_in_left",
            label_visibility="collapsed",
            on_change=_scale_calc_update_real_from_in,
        )
        in_sep_col.markdown(":")
        in_right_col.number_input(
            "IN valor 2",
            min_value=0.0001,
            step=1.0,
            format="%.4f",
            key="scale_calc_in_right",
            label_visibility="collapsed",
            on_change=_scale_calc_update_real_from_in,
        )
        in_unit_col.selectbox(
            "IN unidade",
            ["m", "cm", "mm"],
            key="scale_calc_in_unit",
            label_visibility="collapsed",
            on_change=_scale_calc_update_real_from_in,
        )

        st.caption("OUT ficheiro exportado DXF")
        out_left_col, out_sep_col, out_right_col, out_unit_col = st.columns([1.0, 0.15, 1.0, 0.75])
        out_left_col.number_input(
            "OUT valor 1",
            min_value=0.0001,
            step=1.0,
            format="%.4f",
            key="scale_calc_out_left",
            label_visibility="collapsed",
            on_change=_scale_calc_update_model_from_real_or_out,
        )
        out_sep_col.markdown(":")
        out_right_col.number_input(
            "OUT valor 2",
            min_value=0.0001,
            step=50.0,
            format="%.4f",
            key="scale_calc_out_right",
            label_visibility="collapsed",
            on_change=_scale_calc_update_model_from_real_or_out,
        )
        out_unit_col.selectbox(
            "OUT unidade",
            ["m", "cm", "mm"],
            key="scale_calc_out_unit",
            label_visibility="collapsed",
        )

        st.caption("Corresponde a medida")
        real_col, real_unit_col, arrow_col, model_col, model_unit_col = st.columns([1.1, 0.75, 0.25, 1.1, 0.75])
        real_col.number_input(
            "REAL",
            min_value=0.0,
            step=0.5,
            format="%.4f",
            key="scale_calc_real_value",
            on_change=_scale_calc_update_model_from_real_or_out,
        )
        real_unit_col.selectbox(
            "Unidade real",
            ["m", "cm", "mm"],
            key="scale_calc_real_unit",
            label_visibility="collapsed",
            on_change=_scale_calc_update_model_from_real_or_out,
        )
        arrow_col.markdown("->")
        model_col.number_input(
            "MAQUETE",
            min_value=0.0001,
            step=0.5,
            format="%.4f",
            key="scale_calc_model_value",
            on_change=_scale_calc_update_out_from_model,
        )
        model_unit_col.selectbox(
            "Unidade maquete",
            ["mm", "cm", "m"],
            key="scale_calc_model_unit",
            label_visibility="collapsed",
            on_change=_scale_calc_update_out_from_model,
        )

        st.caption(
            f"DXF {st.session_state['scale_calc_file_measure']:.4g} un. -> "
            f"{st.session_state['scale_calc_real_value']:.4g} {st.session_state['scale_calc_real_unit']} reais -> "
            f"{st.session_state['scale_calc_model_value']:.4g} {st.session_state['scale_calc_model_unit']} na maquete"
        )
    if "Metros" in dxf_units:
        units_to_mm = 1000.0
    elif "Centimetros" in dxf_units:
        units_to_mm = 10.0
    else:
        units_to_mm = 1.0
    scale_factor = units_to_mm / scale_denominator
    slicing_mode_choice = st.sidebar.radio(
        "Tipo de Corte da Maquete",
        ["Solido (Opcao A - Empilhamento Cheio)", "Aneis (Opcao B - Vazado/Oco)"],
        index=0,
    )
    slicing_mode = "solid" if "Solido" in slicing_mode_choice else "hollow"
    hollow_glue_margin_mm = 0.0
    if slicing_mode == "hollow":
        hollow_glue_margin_mm = st.sidebar.number_input(
            "Margem montagem/cola (mm)",
            value=0.0,
            min_value=0.0,
            step=0.5,
            help=(
                "No modo Aneis, prolonga cada camada para a zona escondida pela camada acima, "
                "criando base/cola sem alterar a parte visivel do modelo."
            ),
        )

    st.sidebar.subheader("2.2 Nesting e Exportacao")
    col_bed1, col_bed2 = st.sidebar.columns(2)
    bed_w = col_bed1.number_input("Largura cama (mm)", value=720, min_value=1, step=10)
    bed_h = col_bed2.number_input("Altura cama (mm)", value=430, min_value=1, step=10)
    nesting_margin = st.sidebar.number_input(
        "Margem de seguranca (mm)",
        value=5.0,
        min_value=0.0,
        step=0.5,
        help="Usada no limite da cama e no afastamento minimo entre pecas.",
    )
    safety_margin_clearance_mm = st.sidebar.number_input(
        "Folga extra dentro da margem (mm)",
        value=1.0,
        min_value=0.0,
        step=0.1,
        help="Distancia adicional entre as pecas e a margem de seguranca.",
    )
    notch_close_tolerance_mm = st.sidebar.number_input(
        "Fechar reentrancias pequenas (mm)",
        value=0.5,
        min_value=0.0,
        step=0.1,
        help="0 desliga. Valor aplicado ao poligono fechado antes de converter para LINE.",
    )
    linework_simplify_tolerance_mm = st.sidebar.number_input(
        "Simplificar linhas laser (mm)",
        value=EXPORT_LINEWORK_SIMPLIFY_TOLERANCE_MM,
        min_value=0.0,
        step=0.01,
        format="%.3f",
        help=(
            "Remove pontos quase colineares e micro-segmentos no DXF/SVG final. "
            "Use 0 para exportar sem simplificacao extra."
        ),
    )
    nesting_priority_label = st.sidebar.selectbox(
        "Prioridade do nesting",
        options=["Manter formas inteiras", "Maximizar ocupacao das folhas"],
        index=0,
        help=(
            "Maximizar tenta escolher encaixes mais compactos, mas nao parte pecas que cabem inteiras. "
            "So divide uma peca quando ela nao cabe na area util da folha."
        ),
    )
    nesting_priority = (
        "maximize_usage"
        if "Ocupacao" in nesting_priority_label or "ocupacao" in nesting_priority_label.lower()
        else "preserve_shapes"
    )
    target_sheet_utilization = st.sidebar.slider(
        "Percentagem de esforco do nesting (%)",
        min_value=10,
        max_value=100,
        value=65,
        step=5,
        key="nesting_effort_percent",
        help=(
            "Controla quanta exploracao o nesting faz. "
            "Baixo: rapido. Medio: equilibrado. Alto: mais tentativas e mais tempo. "
            "A alteracao e aplicada ao clicar em 'Gerar Folhas de Corte'."
        ),
    )
    split_from_sheet_count = st.sidebar.number_input(
        "Partir pecas a partir de (n folhas)",
        min_value=0,
        value=0,
        step=1,
        help=(
            "0 desliga cortes opcionais. "
            "Se a solucao sem partir gerar pelo menos este numero de folhas, "
            "o nesting pode testar partir algumas pecas para reduzir folhas."
        ),
    )
    preserve_whole_percent = 100
    st.sidebar.caption(
        "Regra base: nao partir pecas que cabem inteiras. "
        "Cortes opcionais so arrancam a partir do limite de folhas configurado."
    )
    st.sidebar.caption(
        "Nota: para pecas grandes no limite, use 'Manter formas inteiras'. "
        "A verificacao usa a area util da folha (cama - margens e folga)."
    )
    assembly_offset_mm = st.sidebar.number_input(
        "Offset gravacao montagem (mm)",
        value=5.0,
        min_value=0.0,
        step=0.5,
    )
    export_units_label = st.sidebar.selectbox(
        "Unidades do documento exportado (DXF/SVG)",
        options=[
            "Automatico (mm)",
            "Milimetros (mm)",
            "Centimetros (cm)",
            "Metros (m)",
        ],
        index=0,
        help="Automatico usa milimetros para manter compatibilidade com o fluxo atual.",
    )
    export_units_map = {
        "Automatico (mm)": "auto",
        "Milimetros (mm)": "mm",
        "Centimetros (cm)": "cm",
        "Metros (m)": "m",
    }
    export_units = export_units_map.get(export_units_label, "auto")

    with st.sidebar.expander("Layers e cores da exportacao", expanded=False):
        st.caption("A cor DXF usa indice ACI (1-255).")

        export_presets = {
            "Personalizado": None,
            "FabLab ISCTE-IUL": {
                "cut": {"name": "CORTE", "dxf_color": 1, "svg_color": "#ff0000"},
                "engrave": {"name": "GRAVAÇÃO", "dxf_color": 5, "svg_color": "#0057ff"},
                "mark": {"name": "MANCHA", "dxf_color": 7, "svg_color": "#000000"},
                "bed": {"name": "Placa", "dxf_color": 8, "svg_color": "#333333"},
                "margin": {"name": "MargemSeguranca", "dxf_color": 8, "svg_color": "#777777"},
            },
            "Trotec (Vermelho/Azul)": {
                "cut": {"name": "CUT", "dxf_color": 1, "svg_color": "#ff0000"},
                "engrave": {"name": "ENGRAVE", "dxf_color": 5, "svg_color": "#0000ff"},
                "mark": {"name": "MARK", "dxf_color": 7, "svg_color": "#000000"},
                "bed": {"name": "BED", "dxf_color": 8, "svg_color": "#333333"},
                "margin": {"name": "SAFE_MARGIN", "dxf_color": 8, "svg_color": "#777777"},
            },
            "Glowforge (Preto/Azul)": {
                "cut": {"name": "CUT", "dxf_color": 7, "svg_color": "#000000"},
                "engrave": {"name": "SCORE", "dxf_color": 5, "svg_color": "#0000ff"},
                "mark": {"name": "FILL", "dxf_color": 8, "svg_color": "#222222"},
                "bed": {"name": "BED", "dxf_color": 8, "svg_color": "#333333"},
                "margin": {"name": "SAFE_MARGIN", "dxf_color": 8, "svg_color": "#777777"},
            },
        }

        preset_name = st.selectbox(
            "Preset de exportacao",
            options=list(export_presets.keys()),
            key="export_preset_name",
            help="Selecione um preset e ajuste manualmente se necessario.",
        )
        reapply_preset = st.button("Aplicar preset", key="apply_export_preset")

        for prefix, defaults in DEFAULT_LAYER_CONFIG.items():
            st.session_state.setdefault(f"layer_name_{prefix}", defaults["name"])
            st.session_state.setdefault(f"layer_aci_{prefix}", int(defaults["dxf_color"]))
            st.session_state.setdefault(f"layer_svg_{prefix}", defaults["svg_color"])

        def _apply_preset_values(preset_key):
            preset_cfg = export_presets.get(preset_key)
            if not preset_cfg:
                return
            for prefix, values in preset_cfg.items():
                st.session_state[f"layer_name_{prefix}"] = values["name"]
                st.session_state[f"layer_aci_{prefix}"] = int(values["dxf_color"])
                st.session_state[f"layer_svg_{prefix}"] = values["svg_color"]

        last_preset = st.session_state.get("export_preset_applied")
        if preset_name != "Personalizado" and last_preset != preset_name:
            _apply_preset_values(preset_name)
            st.session_state["export_preset_applied"] = preset_name
        elif reapply_preset and preset_name != "Personalizado":
            _apply_preset_values(preset_name)
            st.session_state["export_preset_applied"] = preset_name
        elif preset_name == "Personalizado" and last_preset != "Personalizado":
            st.session_state["export_preset_applied"] = "Personalizado"

        def _layer_inputs(prefix, label):
            name_key = f"layer_name_{prefix}"
            aci_key = f"layer_aci_{prefix}"
            svg_key = f"layer_svg_{prefix}"
            name_col, color_col = st.columns([2, 1])
            layer_name = name_col.text_input(
                f"Layer {label}",
                key=name_key,
            )
            dxf_color = int(color_col.number_input(
                f"ACI {label}",
                min_value=1,
                max_value=255,
                step=1,
                key=aci_key,
            ))
            svg_color = st.color_picker(
                f"Cor SVG {label}",
                key=svg_key,
            )
            return {
                "name": layer_name,
                "dxf_color": dxf_color,
                "svg_color": svg_color,
            }

        export_layer_config = {
            "cut": _layer_inputs("cut", "Corte"),
            "engrave": _layer_inputs("engrave", "Gravacao"),
            "mark": _layer_inputs("mark", "Mancha"),
            "bed": _layer_inputs("bed", "Placa"),
            "margin": _layer_inputs("margin", "MargemSeguranca"),
        }

    st.header("3. Camadas")

    topographic_points = st.session_state.get("topographic_points", [])
    selected_points_layer = st.session_state.get("selected_points_layer")
    selected_boundary_layer = st.session_state.get("selected_boundary_layer")

    # Hide boundary outlines only in this preview window.
    boundary_preview_ids = set()
    if active_polygons:
        if selected_boundary_layer is not None:
            boundary_preview_ids = {
                p["id"] for p in active_polygons
                if p["layer"] == selected_boundary_layer
            }
        else:
            named_boundary = [
                p for p in active_polygons
                if p["layer"].lower() == "boundary"
            ]
            if named_boundary:
                boundary_preview_ids = {p["id"] for p in named_boundary}
            else:
                largest = max(active_polygons, key=lambda p: p["geometry"].area)
                boundary_preview_ids = {largest["id"]}

    G = st.session_state["topology"]
    unassigned = [n for n in G.nodes() if G.nodes[n]["data"].get("elevation") is None]
    elevation_values = [
        G.nodes[n]["data"].get("elevation")
        for n in G.nodes()
        if G.nodes[n]["data"].get("elevation") is not None
    ]
    uses_point_elevations = any(
        G.nodes[n]["data"].get("estimated_from_points")
        for n in G.nodes()
    )
    min_elev = min(elevation_values) if elevation_values else None
    max_elev = max(elevation_values) if elevation_values else None

    fig = go.Figure()
    for p in active_polygons:
        poly_id = p["id"]
        if poly_id in boundary_preview_ids:
            continue

        geom = p["geometry"]
        x, y = geom.exterior.xy
        elev = G.nodes[poly_id]["data"].get("elevation") if G.has_node(poly_id) else None

        color = "gray"
        width = 1
        if elev is not None and uses_point_elevations and max_elev != min_elev:
            position = (elev - min_elev) / (max_elev - min_elev)
            color = sample_colorscale("Viridis", position)[0]
        elif elev is not None:
            color = "blue" if elev < 0 else ("red" if elev > 0 else "black")
            if elev == 0:
                color = "green"
                width = 3
        if G.has_node(poly_id) and G.nodes[poly_id]["data"].get("is_descending"):
            color = "orange"
            width = 3

        fig.add_trace(go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines+markers",
            line=dict(color=color, width=width),
            marker=dict(size=6, color="rgba(0,0,0,0)"),
            name=f"ID: {poly_id} Elev: {elev}",
            customdata=[poly_id] * len(x),
        ))

    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        showlegend=False,
        clickmode="event+select",
        dragmode="pan",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Atribuicao de cotas por pontos 3D")
    if topographic_points:
        z_values = [p["z"] for p in topographic_points]
        points_layer_label = selected_points_layer if selected_points_layer else "(nao definida)"
        st.info(
            f"Topografia lida: {len(topographic_points)} pontos 3D | "
            f"Z {min(z_values):.2f} a {max(z_values):.2f}. "
            f"Layer de pontos ativa: {points_layer_label}."
        )
        st.caption(
            "Atribuicao automatica: usa os Z dos pontos 3D detetados sobre cada curva fechada."
        )
        if st.button("Atribuir cotas pelos pontos 3D"):
            st.session_state["topology"] = estimate_polygon_elevations_from_points(
                st.session_state["topology"],
                topographic_points,
            )
            st.rerun()
    else:
        st.info("Sem pontos 3D ativos. Ative uma camada de pontos 3D para atribuir cotas automaticamente.")

    points_manual_overrides = _normalize_override_map(
        st.session_state.get("manual_contour_overrides_points", {})
    )
    summary_manual_overrides = _normalize_override_map(
        st.session_state.get("manual_contour_overrides_summary", {})
    )
    manual_overrides = _resolve_effective_overrides(
        points_manual_overrides,
        summary_manual_overrides,
        st.session_state.get("manual_override_last_source"),
    )
    st.session_state["manual_contour_overrides"] = manual_overrides
    with st.expander("Correcao manual de contornos (sem ficheiros)", expanded=False):
        st.caption(
            "Edite os valores e clique em 'Aplicar alteracoes'. "
            "Tambem pode adicionar linhas novas com o botao '+' da tabela."
        )
        if not topographic_points:
            st.info("Ative uma camada de pontos 3D para gerar a tabela de correcao.")
        else:
            boundary_geom_preview, contour_polygons_preview, _, _ = _resolve_boundary_and_contours(
                active_polygons,
                selected_boundary_layer,
            )
            if boundary_geom_preview is None or not contour_polygons_preview:
                st.info("Sem contornos ativos para correcao.")
            else:
                correction_rows = build_contour_correction_table(
                    contour_polygons_preview,
                    topographic_points,
                    elevation_precision=elevation_group_precision,
                    interval=interval,
                    target_layer_count=target_layer_count,
                )
                if not correction_rows:
                    st.info("Nao foi possivel estimar cotas para montar a tabela.")
                else:
                    min_point_z = min(point["z"] for point in topographic_points)

                    editor_rows = []
                    known_ids = set()
                    # Pre-fill editor with previously saved corrections.
                    for row in correction_rows:
                        contour_key = str(row.get("contour_id", "")).strip()
                        known_ids.add(contour_key)
                        existing = manual_overrides.get(contour_key, {})
                        keep_flag = _parse_bool(existing.get("keep", True), default=True)

                        cota_origem = row.get("estimated_elevation")
                        cota_relativa = row.get("estimated_relative_elevation")
                        camada = row.get("estimated_layer_index")

                        forced_source = _as_float_or_none(existing.get("forced_source_elevation"))
                        if forced_source is None:
                            forced_source = _as_float_or_none(existing.get("forced_elevation"))
                        forced_relative = _as_float_or_none(existing.get("forced_relative_elevation"))
                        forced_layer = _as_int_or_none(existing.get("forced_layer_index"))

                        if forced_source is not None:
                            cota_origem = forced_source
                        if forced_relative is not None:
                            cota_relativa = forced_relative
                        if forced_layer is not None:
                            camada = forced_layer

                        if cota_origem is not None and cota_relativa is None:
                            cota_relativa = cota_origem - min_point_z
                        if cota_relativa is not None and cota_origem is None:
                            cota_origem = min_point_z + cota_relativa
                        if camada is None and cota_relativa is not None and interval > 0:
                            camada = int(round(cota_relativa / interval))

                        editor_rows.append({
                            "contour_id": contour_key,
                            "layer": row.get("layer"),
                            "cota_origem": cota_origem,
                            "cota_relativa": cota_relativa,
                            "camada": camada,
                            "keep": keep_flag,
                        })

                    # Keep user-added manual rows that are not present in auto-detected contours.
                    for contour_key, existing in manual_overrides.items():
                        contour_key = str(contour_key).strip()
                        if contour_key == "" or contour_key in known_ids:
                            continue
                        keep_flag = _parse_bool(existing.get("keep", True), default=True)

                        cota_origem = _as_float_or_none(existing.get("forced_source_elevation"))
                        if cota_origem is None:
                            cota_origem = _as_float_or_none(existing.get("forced_elevation"))
                        cota_relativa = _as_float_or_none(existing.get("forced_relative_elevation"))
                        camada = _as_int_or_none(existing.get("forced_layer_index"))

                        if cota_origem is not None and cota_relativa is None:
                            cota_relativa = cota_origem - min_point_z
                        if cota_relativa is not None and cota_origem is None:
                            cota_origem = min_point_z + cota_relativa
                        if camada is None and cota_relativa is not None and interval > 0:
                            camada = int(round(cota_relativa / interval))

                        editor_rows.append({
                            "contour_id": contour_key,
                            "layer": "",
                            "cota_origem": cota_origem,
                            "cota_relativa": cota_relativa,
                            "camada": camada,
                            "keep": keep_flag,
                        })

                    editor_rows.sort(key=lambda row: contour_id_sort_key(row.get("contour_id", "")))

                    edited_rows = st.data_editor(
                        editor_rows,
                        hide_index=True,
                        use_container_width=True,
                        key="contour_correction_editor",
                        num_rows="dynamic",
                        column_order=[
                            "contour_id",
                            "layer",
                            "cota_origem",
                            "cota_relativa",
                            "camada",
                            "keep",
                        ],
                        column_config={
                            "contour_id": st.column_config.TextColumn("ID", required=True),
                            "layer": st.column_config.Column("Layer"),
                            "cota_origem": st.column_config.NumberColumn(
                                "cota_origem",
                                format="%.3f",
                            ),
                            "cota_relativa": st.column_config.NumberColumn(
                                "cota_relativa",
                                format="%.3f",
                            ),
                            "camada": st.column_config.NumberColumn(
                                "camada",
                                format="%d",
                            ),
                            "keep": st.column_config.CheckboxColumn(
                                "Manter",
                                help="Desative para ignorar este contorno.",
                                default=True,
                            ),
                        },
                        disabled=[],
                    )

                    if hasattr(edited_rows, "to_dict"):
                        edited_rows = edited_rows.to_dict(orient="records")

                    if st.button("Aplicar alteracoes da tabela", key="apply_contour_corrections"):
                        parsed_overrides = {}
                        invalid_values = []
                        for row in edited_rows:
                            contour_id = str(row.get("contour_id", "")).strip()
                            if contour_id == "":
                                continue
                            keep_flag = _parse_bool(row.get("keep", True), default=True)

                            cota_origem = _as_float_or_none(row.get("cota_origem"))
                            cota_relativa = _as_float_or_none(row.get("cota_relativa"))
                            camada = _as_int_or_none(row.get("camada"))

                            raw_cota_origem = str(row.get("cota_origem", "")).strip()
                            raw_cota_relativa = str(row.get("cota_relativa", "")).strip()
                            raw_camada = str(row.get("camada", "")).strip()

                            if raw_cota_origem and raw_cota_origem.lower() not in {"none", "nan"} and cota_origem is None:
                                invalid_values.append(f"{contour_id}:cota_origem")
                            if raw_cota_relativa and raw_cota_relativa.lower() not in {"none", "nan"} and cota_relativa is None:
                                invalid_values.append(f"{contour_id}:cota_relativa")
                            if raw_camada and raw_camada.lower() not in {"none", "nan"} and camada is None:
                                invalid_values.append(f"{contour_id}:camada")

                            override = {}
                            if not keep_flag:
                                override["keep"] = False

                            if cota_origem is not None:
                                override["forced_source_elevation"] = cota_origem
                            elif cota_relativa is not None:
                                override["forced_relative_elevation"] = cota_relativa
                            elif camada is not None and camada >= 0:
                                override["forced_layer_index"] = camada
                            elif camada is not None and camada < 0:
                                invalid_values.append(f"{contour_id}:camada")

                            if override:
                                parsed_overrides[contour_id] = override

                        st.session_state["manual_contour_overrides_points"] = parsed_overrides
                        st.session_state["manual_override_last_source"] = "points"
                        summary_now = _normalize_override_map(
                            st.session_state.get("manual_contour_overrides_summary", {})
                        )
                        manual_overrides = _resolve_effective_overrides(
                            parsed_overrides,
                            summary_now,
                            st.session_state.get("manual_override_last_source"),
                        )
                        st.session_state["manual_contour_overrides"] = manual_overrides

                        if invalid_values:
                            st.warning(
                                "Alguns campos foram ignorados por formato invalido: "
                                + ", ".join(invalid_values[:12])
                                + ("..." if len(invalid_values) > 12 else "")
                            )
                        elif manual_overrides:
                            st.success(f"Correcoes ativas em {len(manual_overrides)} contorno(s).")
                        else:
                            st.info("Nao ha correcoes ativas.")

                    st.caption(
                        "Prioridade interna: cota_origem > cota_relativa > camada. "
                        "Entre tabelas, vence sempre a ultima que aplicou alteracoes."
                    )

                    if st.button("Limpar todas as correcoes manuais", key="clear_contour_corrections"):
                        st.session_state["manual_contour_overrides_points"] = {}
                        st.session_state["manual_contour_overrides_summary"] = {}
                        st.session_state["manual_contour_overrides"] = {}
                        st.session_state["manual_override_last_source"] = "points"
                        st.rerun()

    if unassigned:
        st.warning(
            f"{len(unassigned)} curvas nao tem cota atribuida. "
            "Use pontos 3D ou corrija as cotas na tabela."
        )

    st.markdown("---")
    st.subheader("Gerar Formas")

    can_generate_from_points = bool(topographic_points)
    has_elevations = any(G.nodes[n]["data"].get("elevation") is not None for n in G.nodes())
    has_unassigned = any(G.nodes[n]["data"].get("elevation") is None for n in G.nodes())
    if st.button("Gerar Formas", disabled=(not can_generate_from_points and (not has_elevations or has_unassigned))):
        if can_generate_from_points:
            with st.spinner("A gerar camadas pelas polilinhas originais do DXF..."):
                st.session_state["slices"] = _generate_slices_from_points(
                    active_polygons,
                    selected_boundary_layer,
                    topographic_points,
                    interval,
                    elevation_group_precision,
                    scale_factor,
                    slicing_mode,
                    manual_overrides,
                    target_layer_count,
                    assembly_offset_mm,
                    hollow_glue_margin_mm,
                    point_curve_distance,
                    extend_base_to_zero,
                )
                preview_count = len(st.session_state["slices"])
                base_groups = [
                    item for item in st.session_state["slices"] if item.get("is_zero_base")
                ]
                added_base_count = sum(_safe_repeat_count(item) for item in base_groups)
                terrain_count = preview_count - len(base_groups)
                if target_layer_count and terrain_count != int(target_layer_count):
                    st.warning(
                        f"Camadas topograficas geradas: {terrain_count}. "
                        f"O total definido e {int(target_layer_count)}; "
                        "verifique se existem curvas/alteracoes geometricas suficientes para todos os niveis."
                    )
                    if added_base_count:
                        st.info(
                            f"Base real: {added_base_count} placas agrupadas em "
                            f"{len(base_groups)} camada(s) para manter o preview leve."
                        )
                else:
                    message = f"Camadas geradas: {terrain_count} topograficas."
                    if added_base_count:
                        message += (
                            f" Base real: {added_base_count} placas agrupadas em "
                            f"{len(base_groups)} camada(s) no preview."
                        )
                    st.success(message)
        else:
            with st.spinner("A calcular operacoes booleanas e aplicar escala..."):
                descending_nodes = {
                    node for node in st.session_state["topology"].nodes()
                    if st.session_state["topology"].nodes[node]["data"].get("is_descending")
                }
                st.session_state["slices"] = generate_rings(
                    st.session_state["topology"],
                    engrave_offset=assembly_offset_mm / scale_factor if scale_factor > 0 else assembly_offset_mm,
                    scale_factor=1.0,
                    direction_changes=descending_nodes,
                    slicing_mode=slicing_mode,
                    boundary_layer=selected_boundary_layer if selected_boundary_layer is not None else "boundary",
                    hollow_glue_margin=hollow_glue_margin_mm / scale_factor if scale_factor > 0 else hollow_glue_margin_mm,
                )
                st.success(f"Formas geradas: {len(st.session_state['slices'])} aneis.")

    if slicing_mode == "hollow" and st.session_state.get("slices") is not None:
        current_glue_margin = hollow_glue_margin_mm / scale_factor if scale_factor > 0 else hollow_glue_margin_mm
        current_engraving_offset = assembly_offset_mm / scale_factor if scale_factor > 0 else assembly_offset_mm
        current_boundary_geom, _, _, _ = _resolve_boundary_and_contours(
            active_polygons,
            selected_boundary_layer,
        )
        hollow_geometry_changed = False
        hollow_slices, hollow_converted = _ensure_hollow_slice_geometry(
            st.session_state["slices"],
            current_glue_margin,
            current_boundary_geom,
        )
        if hollow_converted:
            st.session_state["slices"] = hollow_slices
            st.session_state.pop("last_nested_sheets", None)
            st.session_state.pop("last_nested_scale_factor", None)
            hollow_geometry_changed = True

        normalized_slices, support_rule_changed = _normalize_hollow_support_rule(
            st.session_state["slices"],
            current_glue_margin,
        )
        if support_rule_changed:
            st.session_state["slices"] = normalized_slices
            st.session_state.pop("last_nested_sheets", None)
            st.session_state.pop("last_nested_scale_factor", None)
            hollow_geometry_changed = True

        refreshed_slices, engraving_changed = _refresh_assembly_engraving_lines(
            st.session_state["slices"],
            current_engraving_offset,
            slicing_mode="hollow",
        )
        if engraving_changed:
            st.session_state["slices"] = refreshed_slices
            st.session_state.pop("last_nested_sheets", None)
            st.session_state.pop("last_nested_scale_factor", None)

    if slicing_mode == "solid" and st.session_state.get("slices") is not None:
        solid_slices, solid_visibility_changed, solid_geometry_changed = _ensure_solid_slice_visibility(
            st.session_state["slices"]
        )
        if solid_visibility_changed:
            st.session_state["slices"] = solid_slices
            if solid_geometry_changed:
                st.session_state.pop("last_nested_sheets", None)
                st.session_state.pop("last_nested_scale_factor", None)

        current_engraving_offset = assembly_offset_mm / scale_factor if scale_factor > 0 else assembly_offset_mm
        refreshed_slices, engraving_changed = _refresh_assembly_engraving_lines(
            st.session_state["slices"],
            current_engraving_offset,
            slicing_mode="solid",
        )
        if engraving_changed:
            st.session_state["slices"] = refreshed_slices
            st.session_state.pop("last_nested_sheets", None)
            st.session_state.pop("last_nested_scale_factor", None)

    if st.session_state.get("slices") is not None:
        st.subheader("Preview das Camadas Geradas")
        slice_rows = []
        slice_bounds = []
        for idx, geom_dict in enumerate(st.session_state["slices"]):
            geom = geom_dict["slice_geom"]
            if geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            slice_bounds.append((minx, miny, maxx, maxy))
            slice_rows.append({
                "_row_id": idx,
                "_source_ids": geom_dict.get("source_ids") or [],
                "camada": idx,
                "quantidade": _safe_repeat_count(geom_dict),
                "id": geom_dict["id"],
                "cota_relativa": geom_dict.get("elevation"),
                "cota_origem": geom_dict.get("source_elevation"),
                "area": round(geom.area, 2),
                "area_removida": geom_dict.get("removed_area"),
                "curvas": ", ".join(str(cid) for cid in (geom_dict.get("source_ids") or [])),
            })

        if slice_rows:
            with st.expander("Resumo das camadas"):
                edited_slice_rows = st.data_editor(
                    slice_rows,
                    hide_index=True,
                    use_container_width=True,
                    key="slice_summary_editor",
                    num_rows="dynamic",
                    column_order=[
                        "camada",
                        "quantidade",
                        "id",
                        "cota_relativa",
                        "cota_origem",
                        "area",
                        "area_removida",
                        "curvas",
                    ],
                    column_config={
                        "camada": st.column_config.NumberColumn("camada", format="%d"),
                        "quantidade": st.column_config.NumberColumn("quantidade", format="%d"),
                        "id": st.column_config.Column("id"),
                        "cota_relativa": st.column_config.NumberColumn(
                            "cota_relativa",
                            format="%.3f",
                        ),
                        "cota_origem": st.column_config.NumberColumn(
                            "cota_origem",
                            format="%.3f",
                        ),
                        "area": st.column_config.NumberColumn("area", format="%.2f"),
                        "area_removida": st.column_config.NumberColumn("area_removida", format="%.3f"),
                        "curvas": st.column_config.TextColumn("curvas"),
                    },
                    disabled=["id", "quantidade", "area", "area_removida"],
                )

                if hasattr(edited_slice_rows, "to_dict"):
                    edited_slice_rows = edited_slice_rows.to_dict(orient="records")

                st.caption(
                    "Entre as duas tabelas, vence sempre a ultima que aplicou alteracoes."
                )

                if st.button("Aplicar correcoes desta tabela", key="apply_slice_summary_corrections"):
                    summary_overrides = _normalize_override_map(
                        st.session_state.get("manual_contour_overrides_summary", {})
                    )
                    applied_count = 0
                    invalid_rows = []

                    original_by_row = {row["_row_id"]: row for row in slice_rows}
                    for row in edited_slice_rows:
                        row_id = row.get("_row_id")
                        original = original_by_row.get(row_id)
                        if original is None:
                            source_ids = _parse_curve_ids(row.get("curvas"))
                            if source_ids is None:
                                invalid_rows.append("curvas")
                                continue
                        else:
                            parsed_from_text = _parse_curve_ids(row.get("curvas"))
                            if parsed_from_text is None:
                                invalid_rows.append(str(original.get("camada")))
                                continue
                            source_ids = parsed_from_text or (original.get("_source_ids") or [])

                        if not source_ids:
                            continue

                        forced_source = _as_float_or_none(row.get("cota_origem"))
                        forced_relative = _as_float_or_none(row.get("cota_relativa"))
                        forced_layer = _as_int_or_none(row.get("camada"))

                        if original is None:
                            changed_source = forced_source is not None
                            changed_relative = forced_relative is not None
                            changed_layer = forced_layer is not None
                        else:
                            original_source = _as_float_or_none(original.get("cota_origem"))
                            original_relative = _as_float_or_none(original.get("cota_relativa"))
                            original_layer = _as_int_or_none(original.get("camada"))

                            changed_source = (
                                forced_source is not None
                                and (
                                    original_source is None
                                    or abs(forced_source - original_source) > 1e-9
                                )
                            )
                            changed_relative = (
                                forced_relative is not None
                                and (
                                    original_relative is None
                                    or abs(forced_relative - original_relative) > 1e-9
                                )
                            )
                            changed_layer = (
                                forced_layer is not None
                                and (
                                    original_layer is None
                                    or forced_layer != original_layer
                                )
                            )

                        if not (changed_source or changed_relative or changed_layer):
                            continue

                        for contour_id in source_ids:
                            key = str(contour_id)
                            override = dict(summary_overrides.get(key, {}))
                            if changed_source:
                                override["forced_source_elevation"] = forced_source
                            if changed_relative:
                                override["forced_relative_elevation"] = forced_relative
                            if changed_layer:
                                override["forced_layer_index"] = forced_layer
                            summary_overrides[key] = override
                            applied_count += 1

                    st.session_state["manual_contour_overrides_summary"] = summary_overrides
                    st.session_state["manual_override_last_source"] = "summary"
                    points_overrides_now = _normalize_override_map(
                        st.session_state.get("manual_contour_overrides_points", {})
                    )
                    effective_overrides = _resolve_effective_overrides(
                        points_overrides_now,
                        summary_overrides,
                        st.session_state.get("manual_override_last_source"),
                    )
                    st.session_state["manual_contour_overrides"] = effective_overrides
                    manual_overrides = effective_overrides

                    if applied_count > 0:
                        st.success(
                            f"Correcoes aplicadas a {applied_count} atribuicao(oes) de curva."
                        )
                        if invalid_rows:
                            st.warning("Algumas linhas foram ignoradas por 'curvas' invalido.")

                        # Refresh layers immediately using the latest applied table.
                        if topographic_points:
                            with st.spinner("A aplicar correcoes mais recentes e atualizar camadas..."):
                                st.session_state["slices"] = _generate_slices_from_points(
                                    active_polygons,
                                    selected_boundary_layer,
                                    topographic_points,
                                    interval,
                                    elevation_group_precision,
                                    scale_factor,
                                    slicing_mode,
                                    effective_overrides,
                                    target_layer_count,
                                    assembly_offset_mm,
                                    hollow_glue_margin_mm,
                                    point_curve_distance,
                                    extend_base_to_zero,
                                )
                            st.rerun()
                    elif invalid_rows:
                        st.warning("Existem linhas com 'curvas' invalido. Use IDs separados por virgula.")
                    else:
                        st.info("Sem alteracoes novas para aplicar.")

        if slice_bounds:
            global_minx = min(bounds[0] for bounds in slice_bounds)
            global_miny = min(bounds[1] for bounds in slice_bounds)
            global_maxx = max(bounds[2] for bounds in slice_bounds)
            global_maxy = max(bounds[3] for bounds in slice_bounds)
            pad = max(global_maxx - global_minx, global_maxy - global_miny) * 0.05
            x_range = [global_minx - pad, global_maxx + pad]
            y_range = [global_miny - pad, global_maxy + pad]
        else:
            x_range = None
            y_range = None

        slice_preview_container = st.expander(
            "Visualizacao das camadas geradas",
            expanded=False,
        )
        layer_preview_grid_columns = _preview_grid_columns_state("layer_preview_grid_columns")
        layer_preview_grid_columns = slice_preview_container.radio(
            "Janelas por linha",
            options=[1, 2, 3, 4],
            horizontal=True,
            key="layer_preview_grid_columns",
            help="Altera apenas o layout desta visualizacao.",
        )
        layer_preview_grid_columns = max(1, min(4, int(layer_preview_grid_columns)))
        layer_preview_height = _preview_height_for_columns(
            layer_preview_grid_columns,
            LAYER_PREVIEW_HEIGHT_BY_COLUMNS,
            LAYER_PREVIEW_HEIGHT,
        )
        slice_preview_columns = []
        visible_slice_index = 0

        def _polygon_parts_2d(geom):
            if geom is None or geom.is_empty:
                return []
            if geom.geom_type == "Polygon":
                return [geom]
            if geom.geom_type == "MultiPolygon":
                return list(geom.geoms)
            return []

        def _add_geometry_trace(fig, geom, color, width, name, dash=None, fill=None):
            legend_added = False
            for poly in _polygon_parts_2d(geom):
                x, y = poly.exterior.xy
                fig.add_trace(go.Scatter(
                    x=list(x),
                    y=list(y),
                    mode="lines",
                    fill=fill,
                    fillcolor="rgba(245, 158, 11, 0.18)" if fill else None,
                    line=dict(color=color, width=width, dash=dash),
                    name=name,
                    showlegend=not legend_added,
                ))
                legend_added = True
                for interior in poly.interiors:
                    ix, iy = interior.xy
                    fig.add_trace(go.Scatter(
                        x=list(ix),
                        y=list(iy),
                        mode="lines",
                        line=dict(color=color, width=max(1, width - 0.5), dash=dash),
                        name=name,
                        showlegend=False,
                    ))

        for idx, geom_dict in enumerate(st.session_state["slices"]):
            slice_geom = geom_dict["slice_geom"]
            visible_geom = geom_dict.get("visible_geom", slice_geom)
            support_geom = geom_dict.get("support_geom")
            fig_slice = go.Figure()
            show_layer_roles = slicing_mode == "hollow"
            if show_layer_roles:
                _add_geometry_trace(fig_slice, support_geom, "#f59e0b", LAYER_PREVIEW_ROLE_LINE_WIDTH, "base/cola", dash="dot", fill="toself")
            _add_geometry_trace(fig_slice, slice_geom, "#ef4444", LAYER_PREVIEW_CUT_LINE_WIDTH, "corte")
            if show_layer_roles:
                _add_geometry_trace(fig_slice, visible_geom, "#2563eb", LAYER_PREVIEW_ROLE_LINE_WIDTH, "visivel", dash="dash")

            if _polygon_parts_2d(slice_geom):
                source_elev = geom_dict.get("source_elevation")
                relative_elev = geom_dict.get("elevation")
                title = f"Camada {idx} | cota relativa {relative_elev} m"
                if source_elev is not None:
                    title += f" | Z {source_elev} m"
                repeat_count = _safe_repeat_count(geom_dict)
                if repeat_count > 1:
                    title += f" | quantidade x{repeat_count}"
                title += f" | area {slice_geom.area:.1f}"
                if show_layer_roles and support_geom is not None and not support_geom.is_empty:
                    title += f" | visivel {visible_geom.area:.1f} | base {support_geom.area:.1f}"
                fig_slice.update_layout(
                    title=title,
                    height=layer_preview_height,
                    margin=dict(l=8, r=8, t=70, b=8),
                    title_font=dict(size=13),
                    dragmode="pan",
                    xaxis=dict(scaleanchor="y", scaleratio=1, range=x_range),
                    yaxis=dict(range=y_range),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.01,
                        xanchor="right",
                        x=1,
                        font=dict(size=9),
                    ),
                )
                if visible_slice_index % layer_preview_grid_columns == 0:
                    slice_preview_columns = slice_preview_container.columns(layer_preview_grid_columns)
                with slice_preview_columns[visible_slice_index % layer_preview_grid_columns]:
                    st.plotly_chart(
                        fig_slice,
                        use_container_width=True,
                        key=f"slices_preview_{idx}",
                        config={"displayModeBar": True, "scrollZoom": True},
                    )
                visible_slice_index += 1

    st.markdown("---")
    st.header("4. Folhas e Nesting")
    st.markdown("Gere as folhas de corte e exporte todas em DXF e SVG.")

    if st.session_state.get("slices") is not None and st.button("Gerar Folhas de Corte", type="primary"):
        with st.spinner("A otimizar espaco nas folhas de corte (Nesting)..."):
            output_slices = _scale_slices_for_output(
                st.session_state["slices"],
                scale_factor,
            )
            effective_nesting_margin = nesting_margin + safety_margin_clearance_mm
            sheets = perform_nesting(
                output_slices,
                bed_w=bed_w,
                bed_h=bed_h,
                margin=effective_nesting_margin,
                priority=nesting_priority,
                target_utilization=target_sheet_utilization,
                preserve_whole_percent=preserve_whole_percent,
                split_from_sheet_count=split_from_sheet_count,
            )
            st.session_state["last_nested_sheets"] = sheets
            st.session_state["last_nested_scale_factor"] = scale_factor
        nested_pieces = [piece for sheet in sheets.values() for piece in sheet]
        total_nested_area = sum(piece["geom"].area for piece in nested_pieces)
        split_piece_count = sum(1 for piece in nested_pieces if piece.get("was_split"))
        split_source_count = len({
            piece.get("source_slice_id", piece.get("id"))
            for piece in nested_pieces
            if piece.get("was_split")
        })
        required_split_sources = {
            piece.get("source_slice_id", piece.get("id"))
            for piece in nested_pieces
            if piece.get("was_split") and piece.get("split_reason") == "required"
        }
        optional_split_sources = {
            piece.get("source_slice_id", piece.get("id"))
            for piece in nested_pieces
            if piece.get("was_split") and piece.get("split_reason") != "required"
        }
        usable_sheet_area = max(
            (bed_w - 2 * effective_nesting_margin)
            * (bed_h - 2 * effective_nesting_margin),
            1.0,
        )
        utilization = 100.0 * total_nested_area / max(len(sheets) * usable_sheet_area, 1.0)
        st.success(
            f"Nesting concluido! Foram geradas {len(sheets)} folhas de corte. "
            f"Ocupacao media aprox.: {utilization:.1f}%."
        )
        if required_split_sources:
            st.warning(
                f"Foram cortadas {len(required_split_sources)} peca(s) porque o programa nao conseguiu "
                "faze-las caber inteiras na area util com a margem/folga definida."
            )
        if optional_split_sources:
            st.info(
                f"Foram testados cortes opcionais em {len(optional_split_sources)} peca(s) "
                f"porque a solucao base atingiu o limite de folhas ({int(split_from_sheet_count)})."
            )
        elif nesting_priority == "maximize_usage":
            st.info("Prioridade aplicada: encaixe compacto sem partir pecas que cabem inteiras.")
        else:
            st.info("Prioridade aplicada: todas as pecas foram mantidas inteiras.")

        with st.spinner("A exportar folhas em DXF/SVG..."):
            svgs = export_sheets_to_svg(
                sheets,
                bed_w,
                bed_h,
                margin=nesting_margin,
                layer_config=export_layer_config,
                export_units=export_units,
                geometry_clean_tolerance=notch_close_tolerance_mm,
                linework_simplify_tolerance=linework_simplify_tolerance_mm,
                guide_clearance=safety_margin_clearance_mm,
            )
            dxfs = export_sheets_to_dxf(
                sheets,
                bed_w,
                bed_h,
                margin=nesting_margin,
                layer_config=export_layer_config,
                export_units=export_units,
                geometry_clean_tolerance=notch_close_tolerance_mm,
                linework_simplify_tolerance=linework_simplify_tolerance_mm,
                guide_clearance=safety_margin_clearance_mm,
            )
            combined_dxf = export_sheets_to_combined_dxf(
                sheets,
                bed_w,
                bed_h,
                margin=nesting_margin,
                layer_config=export_layer_config,
                export_units=export_units,
                geometry_clean_tolerance=notch_close_tolerance_mm,
                linework_simplify_tolerance=linework_simplify_tolerance_mm,
                guide_clearance=safety_margin_clearance_mm,
            )

        with st.spinner("A verificar DXF exportado..."):
            export_diagnostics = {
                sid: diagnose_exported_dxf(
                    dxf_content,
                    layer_config=export_layer_config,
                )
                for sid, dxf_content in dxfs.items()
            }
        failed_export_diagnostics = {
            sid: diagnostics
            for sid, diagnostics in export_diagnostics.items()
            if not diagnostics.get("ok")
        }
        status_rows = []
        issue_rows = []
        for sid, diagnostics in export_diagnostics.items():
            first_issue = (diagnostics.get("issues") or [{}])[0]
            status_rows.append(
                {
                    "Folha": sid + 1,
                    "Estado": "OK" if diagnostics.get("ok") else "Verificar",
                    "Entidades": diagnostics.get("entity_count", 0),
                    "Tipos": ", ".join(
                        f"{name}: {count}"
                        for name, count in sorted(diagnostics.get("entity_types", {}).items())
                    ),
                    "Layers": ", ".join(diagnostics.get("layers", [])),
                    "Sobreposicoes": diagnostics.get("overlap_count", 0),
                    "Problemas": len(diagnostics.get("issues", [])),
                    "Primeiro problema": first_issue.get("message", ""),
                    "Layer problema": first_issue.get("layer", ""),
                    "Outra layer": first_issue.get("other_layer", ""),
                }
            )
            for issue in diagnostics.get("issues", []):
                issue_rows.append(
                    {
                        "Folha": sid + 1,
                        "Tipo": issue.get("type", ""),
                        "Layer": issue.get("layer", ""),
                        "Outra layer": issue.get("other_layer", ""),
                        "Mensagem": issue.get("message", ""),
                    }
                )

        if failed_export_diagnostics:
            st.warning(
                "Diagnostico DXF: foram encontrados problemas nas folhas exportadas. "
                "Abra o relatorio abaixo antes de enviar para corte."
            )
        else:
            st.success(
                "Diagnostico DXF: OK. Tudo convertido para LINE, entidades em layers, "
                "propriedades ByLayer e sem linhas sobrepostas detectadas."
            )

        with st.expander("Diagnostico DXF das folhas", expanded=bool(failed_export_diagnostics)):
            st.dataframe(status_rows, use_container_width=True, hide_index=True)
            if issue_rows:
                st.dataframe(issue_rows, use_container_width=True, hide_index=True)
            else:
                st.caption("Sem problemas detectados no DXF exportado.")

        dxf_zip = io.BytesIO()
        with zipfile.ZipFile(dxf_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for sid, dxf_content in dxfs.items():
                archive.writestr(f"folha_corte_{sid + 1}.dxf", dxf_content)

        svg_zip = io.BytesIO()
        with zipfile.ZipFile(svg_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for sid, svg_content in svgs.items():
                archive.writestr(f"folha_corte_{sid + 1}.svg", svg_content)

        st.download_button(
            label="Baixar todas as folhas (DXF)",
            data=dxf_zip.getvalue(),
            file_name="folhas_corte_dxf.zip",
            mime="application/zip",
        )
        st.download_button(
            label="Baixar DXF unico com todas as folhas",
            data=combined_dxf,
            file_name="folhas_corte_todas.dxf",
            mime="application/dxf",
        )
        st.download_button(
            label="Baixar todas as folhas (SVG)",
            data=svg_zip.getvalue(),
            file_name="folhas_corte_svg.zip",
            mime="application/zip",
        )
        with st.expander("Visualizacao das folhas geradas", expanded=False):
            sheet_preview_grid_columns = _preview_grid_columns_state("sheet_preview_grid_columns")
            sheet_preview_grid_columns = st.radio(
                "Janelas por linha",
                options=[1, 2, 3, 4],
                horizontal=True,
                key="sheet_preview_grid_columns",
                help="Altera apenas o layout desta visualizacao.",
            )
            sheet_preview_grid_columns = max(1, min(4, int(sheet_preview_grid_columns)))
            sheet_preview_height = _preview_height_for_columns(
                sheet_preview_grid_columns,
                SHEET_PREVIEW_HEIGHT_BY_COLUMNS,
                SHEET_PREVIEW_HEIGHT,
            )
            sheet_preview_columns = []
            for preview_index, (sid, svg_content) in enumerate(svgs.items()):
                if preview_index % sheet_preview_grid_columns == 0:
                    sheet_preview_columns = st.columns(sheet_preview_grid_columns)
                with sheet_preview_columns[preview_index % sheet_preview_grid_columns]:
                    st.subheader(f"Folha {sid + 1}")
                    b64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
                    st.markdown(
                        (
                            f'<div style="height:{sheet_preview_height}px; width:100%; '
                            'display:flex; align-items:center; justify-content:center; '
                            'border:1px solid #d1d5db; background:#fff; overflow:hidden;">'
                            f'<img src="data:image/svg+xml;base64,{b64}" '
                            'style="width:100%; height:100%; object-fit:contain;"/>'
                            '</div>'
                        ),
                        unsafe_allow_html=True,
                    )
                    dxf_col, svg_col = st.columns(2)
                    with dxf_col:
                        st.download_button(
                            label="Baixar DXF",
                            data=dxfs[sid],
                            file_name=f"folha_corte_{sid + 1}.dxf",
                            mime="application/dxf",
                            key=f"download_sheet_{sid}_dxf",
                            use_container_width=True,
                        )
                    with svg_col:
                        st.download_button(
                            label="Baixar SVG",
                            data=svg_content,
                            file_name=f"folha_corte_{sid + 1}.svg",
                            mime="image/svg+xml",
                            key=f"download_sheet_{sid}_svg",
                            use_container_width=True,
                        )

    if st.session_state.get("slices") is not None:
        st.markdown("---")
        st.header("5. Visualizacao")
        st.markdown("Preview 3D simples das pecas empilhadas.")

        col_v1, col_v2, col_v3 = st.columns(3)
        layer_thickness_mm = col_v1.number_input(
            "Espessura por fatia (mm)",
            value=1.0,
            min_value=0.1,
            step=0.1,
        )
        mesh_opacity = col_v2.slider(
            "Opacidade",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.05,
        )
        visual_style = col_v3.selectbox(
            "Estilo visual",
            ["Cartao branco", "Altimetria colorida"],
            index=0,
        )
        show_contours = st.checkbox("Mostrar contornos", value=True)
        show_split_cuts = st.checkbox("Mostrar cortes das placas divididas", value=True)

        def _polygon_parts_for_3d(geom):
            if geom.is_empty:
                return []
            if geom.geom_type == "Polygon":
                return [geom]
            if geom.geom_type == "MultiPolygon":
                return list(geom.geoms)
            return []

        def _line_parts_for_3d(geom):
            if geom is None or geom.is_empty:
                return []
            if geom.geom_type in ("LineString", "LinearRing"):
                return [geom]
            if geom.geom_type == "MultiLineString":
                return list(geom.geoms)
            if geom.geom_type == "GeometryCollection":
                lines = []
                for part in geom.geoms:
                    lines.extend(_line_parts_for_3d(part))
                return lines
            return []

        def _scaled_z_for_slice(slice_data, index):
            try:
                relative_elevation = float(slice_data.get("elevation"))
            except (TypeError, ValueError):
                relative_elevation = float(index)
            return relative_elevation * scale_factor

        def _split_cut_lines_for_slice(slice_data, nested_sheets):
            if not nested_sheets:
                return []

            source_id = slice_data.get("id")
            pieces = [
                piece
                for sheet in nested_sheets.values()
                for piece in sheet
                if piece.get("source_slice_id") == source_id
            ]
            if len(pieces) <= 1:
                return []

            original_geom = slice_data["slice_geom"]
            original_boundary_clearance = original_geom.boundary.buffer(0.05)
            cut_lines = []
            for piece in pieces:
                model_geom = piece.get("model_geom")
                if model_geom is None or model_geom.is_empty:
                    continue
                seam = model_geom.boundary.difference(original_boundary_clearance).intersection(original_geom)
                cut_lines.extend(_line_parts_for_3d(seam))

            if not cut_lines:
                return []
            merged = unary_union(cut_lines)
            return [
                line
                for line in _line_parts_for_3d(merged)
                if line.length > 0.5
            ]

        def _merge_mesh_parts(parts):
            x_all, y_all, z_all = [], [], []
            i_all, j_all, k_all = [], [], []
            offset = 0
            for x_vals, y_vals, z_vals, i_vals, j_vals, k_vals in parts:
                x_all.extend(x_vals)
                y_all.extend(y_vals)
                z_all.extend(z_vals)
                i_all.extend(index + offset for index in i_vals)
                j_all.extend(index + offset for index in j_vals)
                k_all.extend(index + offset for index in k_vals)
                offset += len(x_vals)
            return x_all, y_all, z_all, i_all, j_all, k_all

        def _build_polygon_mesh(poly, z0, z1):
            x_vals, y_vals, z_vals = [], [], []
            i_vals, j_vals, k_vals = [], [], []

            def add_vertex(x, y, z):
                x_vals.append(float(x))
                y_vals.append(float(y))
                z_vals.append(float(z))
                return len(x_vals) - 1

            def add_triangle(a, b, c):
                i_vals.append(a)
                j_vals.append(b)
                k_vals.append(c)

            for tri in triangulate(poly):
                if tri.is_empty or tri.area <= 1e-9:
                    continue
                if not poly.covers(tri.representative_point()):
                    continue
                coords = list(tri.exterior.coords)[:-1]
                if len(coords) != 3:
                    continue

                top = [add_vertex(px, py, z1) for px, py in coords]
                add_triangle(top[0], top[1], top[2])

                bottom = [add_vertex(px, py, z0) for px, py in coords]
                add_triangle(bottom[2], bottom[1], bottom[0])

            rings = [poly.exterior] + list(poly.interiors)
            for ring in rings:
                coords = list(ring.coords)
                for idx in range(len(coords) - 1):
                    x1, y1 = coords[idx]
                    x2, y2 = coords[idx + 1]
                    b1 = add_vertex(x1, y1, z0)
                    b2 = add_vertex(x2, y2, z0)
                    t1 = add_vertex(x1, y1, z1)
                    t2 = add_vertex(x2, y2, z1)
                    add_triangle(b1, b2, t2)
                    add_triangle(b1, t2, t1)

            return x_vals, y_vals, z_vals, i_vals, j_vals, k_vals

        fig_3d = go.Figure()
        display_slices = _scale_slices_for_output(st.session_state["slices"], scale_factor)

        def _model_preview_geom(slice_data):
            return slice_data["slice_geom"]

        valid_slices = [s for s in display_slices if not _model_preview_geom(s).is_empty]
        total_slices = len(valid_slices)
        nested_sheets_for_3d = (
            st.session_state.get("last_nested_sheets")
            if abs(float(st.session_state.get("last_nested_scale_factor", scale_factor)) - scale_factor) <= 1e-9
            else None
        )
        slice_z_ranges = []
        for index, slice_data in enumerate(valid_slices):
            z0 = _scaled_z_for_slice(slice_data, index)
            z1 = z0 + layer_thickness_mm
            slice_z_ranges.append((z0, z1))

        if valid_slices:
            preview_bounds = [_model_preview_geom(slice_data).bounds for slice_data in valid_slices]
            preview_minx = min(bounds[0] for bounds in preview_bounds)
            preview_miny = min(bounds[1] for bounds in preview_bounds)
            preview_maxx = max(bounds[2] for bounds in preview_bounds)
            preview_maxy = max(bounds[3] for bounds in preview_bounds)
            preview_w = preview_maxx - preview_minx
            preview_h = preview_maxy - preview_miny
            preview_z = max(z1 for _, z1 in slice_z_ranges) - min(z0 for z0, _ in slice_z_ranges)
            vertical_unit_mm = scale_factor
            effective_vertical_interval = elevation_group_precision if elevation_group_precision > 0 else interval
            expected_thickness_mm = effective_vertical_interval * scale_factor if effective_vertical_interval > 0 else None
            st.caption(
                f"Dimensoes do preview a escala 1:{int(scale_denominator)}: "
                f"X {preview_w:.1f} mm x Y {preview_h:.1f} mm x Z {preview_z:.1f} mm. "
                f"Z usa a cota relativa a escala ({vertical_unit_mm:.3f} mm por unidade vertical) "
                "e a espessura fisica por fatia."
            )
            if expected_thickness_mm and abs(layer_thickness_mm - expected_thickness_mm) > 0.05:
                st.warning(
                    f"Para manter a escala vertical igual a X/Y, a espessura por intervalo devia ser "
                    f"{expected_thickness_mm:.2f} mm. A espessura escolhida e {layer_thickness_mm:.2f} mm."
                )

        for index, slice_data in enumerate(valid_slices):
            geom = _model_preview_geom(slice_data)
            z0, z1 = slice_z_ranges[index]
            if visual_style == "Cartao branco":
                color = "#f4f1e8" if index % 2 == 0 else "#ebe6d9"
            else:
                color = sample_colorscale("Viridis", index / max(total_slices - 1, 1))[0]

            parts = []
            for poly in _polygon_parts_for_3d(geom):
                mesh = _build_polygon_mesh(poly, z0, z1)
                if mesh[3]:
                    parts.append(mesh)

            if not parts:
                continue

            x_vals, y_vals, z_vals, i_vals, j_vals, k_vals = _merge_mesh_parts(parts)
            fig_3d.add_trace(go.Mesh3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                i=i_vals,
                j=j_vals,
                k=k_vals,
                color=color,
                opacity=mesh_opacity,
                name=f"Camada {index}",
                flatshading=False,
                lighting=dict(
                    ambient=1.0,
                    diffuse=0.08,
                    specular=0.0,
                    roughness=1.0,
                    fresnel=0.0,
                ),
                lightposition=dict(x=0, y=0, z=100000),
                hovertemplate=(
                    f"Camada {index}<br>"
                    f"Cota relativa: {slice_data.get('elevation')}<br>"
                    f"Espessura: {layer_thickness_mm:.2f} mm<extra></extra>"
                ),
            ))

            if show_contours:
                for poly in _polygon_parts_for_3d(geom):
                    rings = [poly.exterior] + list(poly.interiors)
                    for ring in rings:
                        coords = list(ring.coords)
                        if len(coords) < 2:
                            continue
                        x_line = [float(x) for x, _ in coords]
                        y_line = [float(y) for _, y in coords]
                        z_line = [z1 + 0.01] * len(coords)
                        fig_3d.add_trace(go.Scatter3d(
                            x=x_line,
                            y=y_line,
                            z=z_line,
                            mode="lines",
                            line=dict(
                                color="#b9b29f" if visual_style == "Cartao branco" else "#2a2a2a",
                                width=2,
                            ),
                            hoverinfo="skip",
                            showlegend=False,
                        ))

            if show_split_cuts:
                for line in _split_cut_lines_for_slice(slice_data, nested_sheets_for_3d):
                    coords = list(line.coords)
                    if len(coords) < 2:
                        continue
                    fig_3d.add_trace(go.Scatter3d(
                        x=[float(x) for x, _ in coords],
                        y=[float(y) for _, y in coords],
                        z=[z1 + 0.08] * len(coords),
                        mode="lines",
                        line=dict(color="#d35400", width=5),
                        hovertemplate=f"Corte da placa<br>Camada {index}<extra></extra>",
                        showlegend=False,
                    ))

        if fig_3d.data:
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)",
                    zaxis_title="Z (mm)",
                    aspectmode="data",
                    bgcolor="#ffffff",
                    dragmode="orbit",
                    camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
                ),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=MODEL_3D_PREVIEW_HEIGHT,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(
                fig_3d,
                use_container_width=True,
                key="slices_3d_preview",
                config={"displayModeBar": True, "scrollZoom": True},
            )
        else:
            st.info("Nao foi possivel gerar o modelo 3D com as geometrias atuais.")
