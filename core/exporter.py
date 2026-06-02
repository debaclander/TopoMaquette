import html
import io
import math
from collections import Counter

import ezdxf
from ezdxf.lldxf.const import DXFStructureError
from shapely.affinity import scale as affinity_scale, translate as affinity_translate
import shapely.geometry as sg
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree


DEFAULT_LAYER_CONFIG = {
    "cut": {"name": "CORTE", "dxf_color": 1, "svg_color": "#ff0000"},
    "engrave": {"name": "GRAVAÇÃO", "dxf_color": 5, "svg_color": "#0057ff"},
    "mark": {"name": "MANCHA", "dxf_color": 7, "svg_color": "#000000"},
    "bed": {"name": "Placa", "dxf_color": 8, "svg_color": "#333333"},
    "margin": {"name": "MargemSeguranca", "dxf_color": 8, "svg_color": "#777777"},
}

TEXT_SIZE = 3.0
TEXT_GAP = 4.0
EXTERNAL_LABEL_GAP = 3.0
TEXT_CLEARANCE = 1.0
EPSILON = 1e-6
MIN_EXPORTED_ENGRAVING_LENGTH = 0.01
GUIDE_CLEARANCE_MM = 1.0
EXPORT_GEOMETRY_CLEAN_TOLERANCE_MM = 0.5
EXPORT_LINEWORK_SIMPLIFY_TOLERANCE_MM = 0.05
SUPPORTED_DXF_LINEWORK = {"LINE"}
MAX_OVERLAP_EXAMPLES = 20


def export_sheets_to_svg(
    sheets,
    bed_w,
    bed_h,
    margin=0.0,
    layer_config=None,
    export_units="auto",
    geometry_clean_tolerance=EXPORT_GEOMETRY_CLEAN_TOLERANCE_MM,
    linework_simplify_tolerance=EXPORT_LINEWORK_SIMPLIFY_TOLERANCE_MM,
    guide_clearance=GUIDE_CLEARANCE_MM,
):
    """
    Exporta o dicionario de folhas para SVG.

    Inclui sempre o retangulo da cama laser e, quando aplicavel, o retangulo
    interno da margem de seguranca.
    """
    svg_outputs = {}
    cfg = _resolve_layer_config(layer_config)
    unit_cfg = _resolve_export_units(export_units)
    scale_factor = unit_cfg["scale_from_mm"]
    svg_unit = unit_cfg["svg_unit"]

    bed_w = float(bed_w) * scale_factor
    bed_h = float(bed_h) * scale_factor
    margin = float(margin) * scale_factor
    margin = _safe_margin(bed_w, bed_h, margin)
    text_size = TEXT_SIZE * scale_factor
    guide_clearance = _safe_nonnegative(guide_clearance, GUIDE_CLEARANCE_MM) * scale_factor
    clean_tolerance = _safe_nonnegative(
        geometry_clean_tolerance,
        EXPORT_GEOMETRY_CLEAN_TOLERANCE_MM,
    ) * scale_factor
    line_tolerance = _safe_nonnegative(
        linework_simplify_tolerance,
        EXPORT_LINEWORK_SIMPLIFY_TOLERANCE_MM,
    ) * scale_factor

    for sheet_id, pieces in sheets.items():
        scaled_pieces = _prepare_pieces_for_export(
            pieces,
            scale_factor,
            clean_tolerance,
            line_tolerance,
        )
        trim_linework = _sheet_non_engrave_linework(scaled_pieces, bed_w, bed_h, margin, cfg)
        guide_clip_geom = _piece_clearance_geometry(scaled_pieces, guide_clearance)
        svg = []
        svg.append(
            f'<svg width="{_fmt(bed_w)}{svg_unit}" height="{_fmt(bed_h)}{svg_unit}" viewBox="0 0 {_fmt(bed_w)} {_fmt(bed_h)}" '
            f'xmlns="http://www.w3.org/2000/svg" style="background:white;border:1px solid #777">'
        )

        svg_linework_by_layer = _guide_linework_by_layer(bed_w, bed_h, margin, cfg)
        svg_label_comments = []

        cut_linework = []
        for piece in scaled_pieces:
            for poly in _polygon_parts(piece["geom"]):
                cut_linework.extend(_linework_from_coords(poly.exterior.coords, close=True, tolerance=line_tolerance))
                for interior in poly.interiors:
                    cut_linework.extend(_linework_from_coords(interior.coords, close=True, tolerance=line_tolerance))
        svg_linework_by_layer.setdefault(cfg["cut"]["name"], []).extend(cut_linework)

        engrave_linework = []
        for piece in scaled_pieces:
            for line in _trim_engravings_against_linework(
                piece.get("engravings", []),
                trim_linework,
                clip_geom=piece["geom"],
                line_tolerance=line_tolerance,
            ):
                for part in _line_parts(line):
                    engrave_linework.extend(_linework_from_coords(part.coords, close=False, tolerance=line_tolerance))

            label = _piece_label(piece)
            svg_label_comments.append(f'<!-- {html.escape(label)} -->')
            label_box = _label_box_for_piece(
                piece,
                label,
                bed_w,
                bed_h,
                [other["geom"] for other in scaled_pieces],
                sheet_margin=margin,
                text_size=text_size,
            )
            for coords in _text_stroke_lines(
                label,
                label_box[0],
                label_box[1],
                y_axis_down=True,
                text_size=text_size,
            ):
                engrave_linework.extend(_linework_from_coords(coords, close=False, tolerance=line_tolerance))
        svg_linework_by_layer.setdefault(cfg["engrave"]["name"], []).extend(engrave_linework)
        final_svg_linework = _finalize_linework_by_layer(
            svg_linework_by_layer,
            cfg,
            guide_clip_geom,
            line_tolerance=line_tolerance,
        )

        for comment in svg_label_comments:
            svg.append(comment)
        for key in ("bed", "margin", "cut", "engrave", "mark"):
            layer = cfg[key]
            svg.append(
                f'<g id="{html.escape(layer["name"])}" '
                f'stroke="{layer["svg_color"]}" fill="none" stroke-width="0.5">'
            )
            _append_svg_linework(svg, final_svg_linework.get(layer["name"], []), tolerance=line_tolerance)
            svg.append("</g>")
        svg.append("</svg>")
        svg_outputs[sheet_id] = "\n".join(svg)

    return svg_outputs


def export_sheets_to_dxf(
    sheets,
    bed_w,
    bed_h,
    margin=0.0,
    layer_config=None,
    export_units="auto",
    geometry_clean_tolerance=EXPORT_GEOMETRY_CLEAN_TOLERANCE_MM,
    linework_simplify_tolerance=EXPORT_LINEWORK_SIMPLIFY_TOLERANCE_MM,
    guide_clearance=GUIDE_CLEARANCE_MM,
):
    """
    Exporta uma DXF por folha segundo configuracao de layers e cores.
    """
    outputs = {}
    cfg = _resolve_layer_config(layer_config)
    unit_cfg = _resolve_export_units(export_units)
    scale_factor = unit_cfg["scale_from_mm"]
    dxf_units = unit_cfg["dxf_units"]

    bed_w = float(bed_w) * scale_factor
    bed_h = float(bed_h) * scale_factor
    margin = float(margin) * scale_factor
    margin = _safe_margin(bed_w, bed_h, margin)
    text_size = TEXT_SIZE * scale_factor
    guide_clearance = _safe_nonnegative(guide_clearance, GUIDE_CLEARANCE_MM) * scale_factor
    clean_tolerance = _safe_nonnegative(
        geometry_clean_tolerance,
        EXPORT_GEOMETRY_CLEAN_TOLERANCE_MM,
    ) * scale_factor
    line_tolerance = _safe_nonnegative(
        linework_simplify_tolerance,
        EXPORT_LINEWORK_SIMPLIFY_TOLERANCE_MM,
    ) * scale_factor

    for sheet_id, pieces in sheets.items():
        scaled_pieces = _prepare_pieces_for_export(
            pieces,
            scale_factor,
            clean_tolerance,
            line_tolerance,
        )
        guide_clip_geom = _piece_clearance_geometry(scaled_pieces, guide_clearance)
        doc = ezdxf.new("R2010")
        doc.units = dxf_units
        _setup_layers(doc, cfg)
        msp = doc.modelspace()

        _write_dxf_sheet(
            msp,
            scaled_pieces,
            bed_w,
            bed_h,
            margin,
            cfg,
            text_size,
            line_tolerance=line_tolerance,
        )

        _dedupe_dxf_modelspace_linework(msp, cfg, guide_clip_geom, line_tolerance=line_tolerance)

        stream = io.StringIO()
        doc.write(stream)
        outputs[sheet_id] = doc.encode(stream.getvalue())

    return outputs


def export_sheets_to_combined_dxf(
    sheets,
    bed_w,
    bed_h,
    margin=0.0,
    layer_config=None,
    export_units="auto",
    geometry_clean_tolerance=EXPORT_GEOMETRY_CLEAN_TOLERANCE_MM,
    linework_simplify_tolerance=EXPORT_LINEWORK_SIMPLIFY_TOLERANCE_MM,
    guide_clearance=GUIDE_CLEARANCE_MM,
    sheet_gap=20.0,
):
    """
    Exporta todas as folhas para um unico DXF, organizadas numa grelha.
    """
    cfg = _resolve_layer_config(layer_config)
    unit_cfg = _resolve_export_units(export_units)
    scale_factor = unit_cfg["scale_from_mm"]
    dxf_units = unit_cfg["dxf_units"]

    bed_w = float(bed_w) * scale_factor
    bed_h = float(bed_h) * scale_factor
    margin = _safe_margin(bed_w, bed_h, float(margin) * scale_factor)
    text_size = TEXT_SIZE * scale_factor
    clean_tolerance = _safe_nonnegative(
        geometry_clean_tolerance,
        EXPORT_GEOMETRY_CLEAN_TOLERANCE_MM,
    ) * scale_factor
    line_tolerance = _safe_nonnegative(
        linework_simplify_tolerance,
        EXPORT_LINEWORK_SIMPLIFY_TOLERANCE_MM,
    ) * scale_factor
    guide_clearance = _safe_nonnegative(guide_clearance, GUIDE_CLEARANCE_MM) * scale_factor
    sheet_gap = _safe_nonnegative(sheet_gap, 20.0) * scale_factor

    doc = ezdxf.new("R2010")
    doc.units = dxf_units
    _setup_layers(doc, cfg)
    msp = doc.modelspace()

    sheet_ids = sorted(sheets)
    if not sheet_ids:
        stream = io.StringIO()
        doc.write(stream)
        return doc.encode(stream.getvalue())

    columns = _combined_sheet_columns(len(sheet_ids), bed_w, bed_h)
    guide_clip_geoms = []
    for order, sheet_id in enumerate(sheet_ids):
        col = order % columns
        row = order // columns
        xoff = col * (bed_w + sheet_gap)
        yoff = row * (bed_h + sheet_gap)
        scaled_pieces = _prepare_pieces_for_export(
            sheets[sheet_id],
            scale_factor,
            clean_tolerance,
            line_tolerance,
        )
        shifted_pieces = [_translate_piece_for_export(piece, xoff, yoff) for piece in scaled_pieces]
        guide_clip_geoms.append(_piece_clearance_geometry(shifted_pieces, guide_clearance))
        _write_dxf_sheet(
            msp,
            shifted_pieces,
            bed_w,
            bed_h,
            margin,
            cfg,
            text_size,
            sheet_origin=(xoff, yoff),
            line_tolerance=line_tolerance,
        )

    guide_clip_geom = unary_union([geom for geom in guide_clip_geoms if not geom.is_empty])
    _dedupe_dxf_modelspace_linework(msp, cfg, guide_clip_geom, line_tolerance=line_tolerance)

    stream = io.StringIO()
    doc.write(stream)
    return doc.encode(stream.getvalue())


def diagnose_exported_dxf(dxf_content, layer_config=None, segment_tolerance=1e-5):
    """
    Verifica se um DXF exportado esta pronto para corte laser:
    entidades LINE, layers esperadas, propriedades ByLayer e segmentos sobrepostos.
    """
    cfg = _resolve_layer_config(layer_config)
    expected_layers = {value["name"] for value in cfg.values()}
    closed_layers = {cfg["cut"]["name"]}

    issues = []
    try:
        if isinstance(dxf_content, bytes):
            dxf_text = dxf_content.decode("utf-8", errors="ignore")
        else:
            dxf_text = str(dxf_content)
        doc = ezdxf.read(io.StringIO(dxf_text))
    except (DXFStructureError, UnicodeDecodeError, ValueError) as exc:
        return {
            "ok": False,
            "entity_count": 0,
            "entity_types": {},
            "layers": [],
            "overlap_count": 0,
            "issues": [
                {
                    "type": "invalid_dxf",
                    "message": f"DXF invalido ou ilegivel: {exc}",
                }
            ],
        }

    defined_layers = {layer.dxf.name for layer in doc.layers}
    for layer_name in sorted(expected_layers - defined_layers):
        issues.append(
            {
                "type": "missing_layer_definition",
                "layer": layer_name,
                "message": f"Layer '{layer_name}' nao esta definida no DXF.",
            }
        )

    entity_types = Counter()
    used_layers = Counter()
    segments = []
    closed_polyline_count = 0

    for entity_index, entity in enumerate(doc.modelspace()):
        entity_type = entity.dxftype()
        entity_types[entity_type] += 1
        layer = getattr(entity.dxf, "layer", "")
        used_layers[layer] += 1

        if entity_type not in SUPPORTED_DXF_LINEWORK:
            issues.append(
                {
                    "type": "unsupported_entity",
                    "entity_type": entity_type,
                    "layer": layer,
                    "message": (
                        f"Entidade {entity_type} na layer '{layer}'. "
                        "O ficheiro exportado deve conter apenas entidades LINE."
                    ),
                }
            )

        if layer not in expected_layers:
            issues.append(
                {
                    "type": "unexpected_layer",
                    "entity_type": entity_type,
                    "layer": layer,
                    "message": f"Entidade em layer inesperada: '{layer}'.",
                }
            )

        _append_bylayer_issues(issues, entity, entity_type, layer)

        if entity_type == "LWPOLYLINE":
            is_closed = _entity_is_closed(entity)
            if is_closed:
                closed_polyline_count += 1
            if layer in closed_layers and not is_closed:
                issues.append(
                    {
                        "type": "open_cut_polyline",
                        "entity_type": entity_type,
                        "layer": layer,
                        "message": f"Polilinha de corte/guia aberta na layer '{layer}'.",
                    }
                )
            coords = [(float(x), float(y)) for x, y in entity.get_points("xy")]
            segments.extend(_segments_from_coords(coords, is_closed, layer, entity_index))
        elif entity_type == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            segments.extend(
                _segments_from_coords(
                    [(float(start.x), float(start.y)), (float(end.x), float(end.y))],
                    False,
                    layer,
                    entity_index,
                )
            )

    issues.extend(_find_open_closed_layer_linework(segments, closed_layers, segment_tolerance))
    overlap_issues = _find_overlapping_segments(segments, segment_tolerance)
    issues.extend(overlap_issues)

    return {
        "ok": len(issues) == 0,
        "entity_count": sum(entity_types.values()),
        "entity_types": dict(entity_types),
        "layers": sorted(used_layers),
        "closed_polyline_count": closed_polyline_count,
        "overlap_count": len(overlap_issues),
        "issues": issues,
    }


def _append_bylayer_issues(issues, entity, entity_type, layer):
    color = getattr(entity.dxf, "color", 256)
    linetype = str(getattr(entity.dxf, "linetype", "BYLAYER") or "BYLAYER").upper()
    lineweight = getattr(entity.dxf, "lineweight", -1)

    try:
        color = int(color)
    except (TypeError, ValueError):
        color = None
    try:
        lineweight = int(lineweight)
    except (TypeError, ValueError):
        lineweight = None

    if color != 256:
        issues.append(
            {
                "type": "not_bylayer",
                "property": "color",
                "value": color,
                "entity_type": entity_type,
                "layer": layer,
                "message": f"Entidade {entity_type} na layer '{layer}' nao esta com cor ByLayer.",
            }
        )
    if linetype != "BYLAYER":
        issues.append(
            {
                "type": "not_bylayer",
                "property": "linetype",
                "value": linetype,
                "entity_type": entity_type,
                "layer": layer,
                "message": f"Entidade {entity_type} na layer '{layer}' nao esta com tipo de linha ByLayer.",
            }
        )
    if lineweight != -1:
        issues.append(
            {
                "type": "not_bylayer",
                "property": "lineweight",
                "value": lineweight,
                "entity_type": entity_type,
                "layer": layer,
                "message": f"Entidade {entity_type} na layer '{layer}' nao esta com peso ByLayer.",
            }
        )


def _entity_is_closed(entity):
    is_closed = getattr(entity, "is_closed", False)
    if callable(is_closed):
        return bool(is_closed())
    return bool(is_closed)


def _segments_from_coords(coords, close, layer, entity_index):
    clean = _clean_coords(coords, close=close)
    if len(clean) < 2:
        return []

    path = list(clean)
    if close and len(path) >= 3:
        path.append(path[0])

    segments = []
    for segment_index, (start, end) in enumerate(zip(path, path[1:])):
        if _same_point(start, end):
            continue
        line = sg.LineString([start, end])
        if line.length <= EPSILON:
            continue
        segments.append(
            {
                "layer": layer,
                "entity_index": entity_index,
                "segment_index": segment_index,
                "start": start,
                "end": end,
                "line": line,
            }
        )
    return segments


def _find_overlapping_segments(segments, tolerance):
    if len(segments) < 2:
        return []

    lines = [segment["line"] for segment in segments]
    tree = STRtree(lines)
    issues = []
    seen_pairs = set()

    for index, line in enumerate(lines):
        for candidate in tree.query(line):
            candidate_index = int(candidate)
            if candidate_index <= index:
                continue
            pair = (index, candidate_index)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            other_line = lines[candidate_index]
            overlap_length = _linear_overlap_length(line.intersection(other_line))
            if overlap_length <= tolerance:
                continue

            first = segments[index]
            second = segments[candidate_index]
            issue_type = (
                "overlapping_segments"
                if first["layer"] == second["layer"]
                else "overlapping_segments_between_layers"
            )
            issues.append(
                {
                    "type": issue_type,
                    "layer": first["layer"],
                    "other_layer": second["layer"],
                    "length": round(float(overlap_length), 6),
                    "start": tuple(round(value, 6) for value in first["start"]),
                    "end": tuple(round(value, 6) for value in first["end"]),
                    "message": _overlap_message(first, second, overlap_length),
                }
            )
            if len(issues) >= MAX_OVERLAP_EXAMPLES:
                return issues

    return issues


def _find_open_closed_layer_linework(segments, closed_layers, tolerance):
    endpoint_counts_by_layer = {}
    for segment in segments:
        layer = segment["layer"]
        if layer not in closed_layers:
            continue
        endpoint_counts = endpoint_counts_by_layer.setdefault(layer, Counter())
        endpoint_counts[_point_key(segment["start"], tolerance)] += 1
        endpoint_counts[_point_key(segment["end"], tolerance)] += 1

    issues = []
    for layer, endpoint_counts in endpoint_counts_by_layer.items():
        open_endpoint_count = sum(count % 2 for count in endpoint_counts.values())
        if open_endpoint_count:
            issues.append(
                {
                    "type": "open_cut_linework",
                    "layer": layer,
                    "open_endpoints": open_endpoint_count,
                    "message": (
                        f"Linework aberta na layer '{layer}': "
                        f"{open_endpoint_count} extremidade(s) sem fecho."
                    ),
                }
            )
    return issues


def _point_key(point, tolerance):
    tolerance = max(float(tolerance or EPSILON), EPSILON)
    return (
        int(round(float(point[0]) / tolerance)),
        int(round(float(point[1]) / tolerance)),
    )


def _linear_overlap_length(geom):
    if geom.is_empty:
        return 0.0
    if geom.geom_type in ("LineString", "LinearRing"):
        return float(geom.length)
    if geom.geom_type == "MultiLineString":
        return sum(float(part.length) for part in geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return sum(_linear_overlap_length(part) for part in geom.geoms)
    return 0.0


def _overlap_message(first, second, overlap_length):
    if first["layer"] == second["layer"]:
        return (
            f"Linha sobreposta na layer '{first['layer']}' "
            f"(comprimento {overlap_length:.3f})."
        )
    return (
        f"Linha sobreposta entre layers '{first['layer']}' e '{second['layer']}' "
        f"(comprimento {overlap_length:.3f})."
    )


def _resolve_layer_config(layer_config):
    cfg = {
        key: value.copy()
        for key, value in DEFAULT_LAYER_CONFIG.items()
    }
    if not layer_config:
        return cfg

    for key, default_value in cfg.items():
        custom = layer_config.get(key, {})
        if not isinstance(custom, dict):
            continue

        name = custom.get("name")
        if name:
            default_value["name"] = str(name)

        dxf_color = custom.get("dxf_color")
        if dxf_color is not None:
            default_value["dxf_color"] = _sanitize_dxf_color(dxf_color, default_value["dxf_color"])

        svg_color = custom.get("svg_color")
        if svg_color:
            default_value["svg_color"] = str(svg_color)

    return cfg


def _resolve_export_units(value):
    raw = "auto" if value is None else str(value).strip().lower()
    aliases = {
        "auto": "auto",
        "automatico": "auto",
        "automático": "auto",
        "automatico (mm)": "auto",
        "automático (mm)": "auto",
        "mm": "mm",
        "milimetros": "mm",
        "milímetros": "mm",
        "milimetros (mm)": "mm",
        "milímetros (mm)": "mm",
        "cm": "cm",
        "centimetros": "cm",
        "centímetros": "cm",
        "centimetros (cm)": "cm",
        "centímetros (cm)": "cm",
        "m": "m",
        "metros": "m",
        "metros (m)": "m",
    }
    unit_key = aliases.get(raw, raw)
    if unit_key not in {"auto", "mm", "cm", "m"}:
        unit_key = "auto"

    if unit_key in {"auto", "mm"}:
        return {
            "scale_from_mm": 1.0,
            "svg_unit": "mm",
            "dxf_units": ezdxf.units.MM,
        }
    if unit_key == "cm":
        return {
            "scale_from_mm": 0.1,
            "svg_unit": "cm",
            "dxf_units": ezdxf.units.CM,
        }
    return {
        "scale_from_mm": 0.001,
        "svg_unit": "m",
        "dxf_units": ezdxf.units.M,
    }


def _sanitize_dxf_color(value, fallback):
    try:
        color = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(1, min(255, color))


def _safe_margin(bed_w, bed_h, margin):
    try:
        margin = float(margin)
    except (TypeError, ValueError):
        margin = 0.0

    margin = max(0.0, margin)
    max_margin = max(min(float(bed_w), float(bed_h)) / 2.0 - EPSILON, 0.0)
    return min(margin, max_margin)


def _safe_nonnegative(value, fallback):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(fallback)
    return max(0.0, number)


def _setup_layers(doc, cfg):
    for layer in cfg.values():
        _ensure_layer(doc, layer["name"], layer["dxf_color"])


def _ensure_layer(doc, layer_name, color):
    if layer_name in doc.layers:
        doc.layers.get(layer_name).dxf.color = color
        return
    doc.layers.new(name=layer_name, dxfattribs={"color": color})


def _append_svg_guides(svg, bed_w, bed_h, margin, cfg):
    svg.append(
        f'<g id="{html.escape(cfg["bed"]["name"])}" '
        f'stroke="{cfg["bed"]["svg_color"]}" fill="none" stroke-width="0.5">'
    )
    _append_svg_rect(svg, 0, 0, bed_w, bed_h)
    svg.append("</g>")

    if bed_w > 2 * margin + EPSILON and bed_h > 2 * margin + EPSILON and margin > EPSILON:
        svg.append(
            f'<g id="{html.escape(cfg["margin"]["name"])}" '
            f'stroke="{cfg["margin"]["svg_color"]}" fill="none" stroke-width="0.5">'
        )
        _append_svg_rect(svg, margin, margin, bed_w - 2 * margin, bed_h - 2 * margin)
        svg.append("</g>")


def _append_dxf_guides(msp, bed_w, bed_h, margin, cfg, offset=(0.0, 0.0)):
    xoff, yoff = offset
    _add_dxf_rect(msp, xoff, yoff, bed_w, bed_h, cfg["bed"]["name"])
    if bed_w > 2 * margin + EPSILON and bed_h > 2 * margin + EPSILON and margin > EPSILON:
        _add_dxf_rect(
            msp,
            xoff + margin,
            yoff + margin,
            bed_w - 2 * margin,
            bed_h - 2 * margin,
            cfg["margin"]["name"],
        )


def _write_dxf_sheet(
    msp,
    pieces,
    bed_w,
    bed_h,
    margin,
    cfg,
    text_size,
    sheet_origin=(0.0, 0.0),
    line_tolerance=0.0,
):
    _append_dxf_guides(msp, bed_w, bed_h, margin, cfg, offset=sheet_origin)
    trim_linework = _sheet_non_engrave_linework(
        pieces,
        bed_w,
        bed_h,
        margin,
        cfg,
        offset=sheet_origin,
    )

    for piece in pieces:
        for poly in _polygon_parts(piece["geom"]):
            _add_dxf_linework(
                msp,
                poly.exterior.coords,
                cfg["cut"]["name"],
                close=True,
                tolerance=line_tolerance,
            )
            for interior in poly.interiors:
                _add_dxf_linework(
                    msp,
                    interior.coords,
                    cfg["cut"]["name"],
                    close=True,
                    tolerance=line_tolerance,
                )

    for piece in pieces:
        for line in _trim_engravings_against_linework(
            piece.get("engravings", []),
            trim_linework,
            clip_geom=piece["geom"],
            line_tolerance=line_tolerance,
        ):
            for part in _line_parts(line):
                _add_dxf_linework(
                    msp,
                    part.coords,
                    cfg["engrave"]["name"],
                    close=False,
                    tolerance=line_tolerance,
                )

        label = _piece_label(piece)
        label_box = _label_box_for_piece(
            piece,
            label,
            bed_w,
            bed_h,
            [other["geom"] for other in pieces],
            sheet_margin=margin,
            text_size=text_size,
            sheet_origin=sheet_origin,
        )
        for coords in _text_stroke_lines(
            label,
            label_box[0],
            label_box[1],
            y_axis_down=False,
            text_size=text_size,
        ):
            _add_dxf_linework(
                msp,
                coords,
                cfg["engrave"]["name"],
                close=False,
                tolerance=line_tolerance,
            )


def _sheet_non_engrave_linework(pieces, bed_w, bed_h, margin, cfg, offset=(0.0, 0.0)):
    lines = []
    for layer_lines in _guide_linework_by_layer(bed_w, bed_h, margin, cfg, offset=offset).values():
        lines.extend(layer_lines)

    for piece in pieces:
        for poly in _polygon_parts(piece["geom"]):
            lines.append(sg.LineString(poly.exterior.coords))
            for interior in poly.interiors:
                lines.append(sg.LineString(interior.coords))

    lines = [line for line in lines if not line.is_empty and line.length > EPSILON]
    if not lines:
        return sg.GeometryCollection()
    return unary_union(lines)


def _piece_clearance_geometry(pieces, clearance):
    geoms = []
    for piece in pieces:
        geom = piece.get("geom")
        if geom is None or geom.is_empty:
            continue
        if clearance > EPSILON:
            geom = geom.buffer(clearance, join_style=2)
        geoms.append(geom)

    if not geoms:
        return sg.GeometryCollection()
    return unary_union(geoms)


def _guide_linework_by_layer(bed_w, bed_h, margin, cfg, offset=(0.0, 0.0)):
    xoff, yoff = offset
    lines_by_layer = {
        cfg["bed"]["name"]: _rect_linework(xoff, yoff, bed_w, bed_h),
    }
    if bed_w > 2 * margin + EPSILON and bed_h > 2 * margin + EPSILON and margin > EPSILON:
        lines_by_layer[cfg["margin"]["name"]] = _rect_linework(
            xoff + margin,
            yoff + margin,
            bed_w - 2 * margin,
            bed_h - 2 * margin,
        )
    return lines_by_layer


def _rect_linework(x, y, width, height):
    coords = _rect_coords(x, y, width, height)
    return [
        sg.LineString([coords[index], coords[(index + 1) % len(coords)]])
        for index in range(len(coords))
    ]


def _trim_engravings_against_linework(
    engravings,
    trim_linework,
    clip_geom=None,
    line_tolerance=0.0,
):
    if not engravings:
        return []

    trimmed_lines = []
    for engraving in engravings:
        for part in _line_parts(engraving):
            source = part
            if clip_geom is not None and not clip_geom.is_empty:
                source = source.intersection(clip_geom)
            if source.is_empty:
                continue
            if trim_linework is not None and not trim_linework.is_empty:
                source = source.difference(trim_linework)
            for line in _line_parts(source):
                for clean_line in _line_parts(_clean_export_linework(line, line_tolerance)):
                    if clean_line.length >= MIN_EXPORTED_ENGRAVING_LENGTH:
                        trimmed_lines.append(clean_line)
    return trimmed_lines


def _append_svg_rect(svg, x, y, width, height):
    coords = _rect_coords(x, y, width, height)
    _append_svg_polyline(svg, coords, close=True)


def _add_dxf_rect(msp, x, y, width, height, layer):
    coords = _rect_coords(x, y, width, height)
    _add_dxf_linework(msp, coords, layer, close=True)


def _rect_coords(x, y, width, height):
    return [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    ]


def _append_svg_polyline(svg, coords, close):
    _append_svg_linework(svg, _linework_from_coords(coords, close))


def _append_svg_linework(svg, lines, tolerance=0.0):
    for line in lines:
        clean = _clean_coords(line.coords, close=False, collinear_tolerance=tolerance)
        for start, end in zip(clean, clean[1:]):
            if _same_point(start, end):
                continue
            svg.append(
                f'<line x1="{_fmt(start[0])}" y1="{_fmt(start[1])}" '
                f'x2="{_fmt(end[0])}" y2="{_fmt(end[1])}" />'
            )


def _linework_from_coords(coords, close, tolerance=0.0):
    clean = _clean_coords(coords, close=close, collinear_tolerance=tolerance)
    if len(clean) < (3 if close else 2):
        return []

    path = list(clean)
    if close:
        path.append(path[0])

    lines = []
    for start, end in zip(path, path[1:]):
        if _same_point(start, end):
            continue
        line = sg.LineString([start, end])
        if line.length > EPSILON:
            lines.append(line)
    return lines


def _add_dxf_linework(msp, coords, layer, close, tolerance=0.0):
    for line in _linework_from_coords(coords, close, tolerance=tolerance):
        coords = _clean_coords(line.coords, close=False, collinear_tolerance=tolerance)
        if len(coords) < 2:
            continue
        start, end = coords[0], coords[-1]
        _add_dxf_line_entity(msp, start, end, layer)


def _dedupe_dxf_modelspace_linework(msp, cfg, guide_clip_geom=None, line_tolerance=0.0):
    lines_by_layer = {}
    for entity in list(msp.query("LINE")):
        start = entity.dxf.start
        end = entity.dxf.end
        line = sg.LineString([(float(start.x), float(start.y)), (float(end.x), float(end.y))])
        if line.length > EPSILON:
            lines_by_layer.setdefault(entity.dxf.layer, []).append(line)
        msp.delete_entity(entity)

    final_linework = _finalize_linework_by_layer(
        lines_by_layer,
        cfg,
        guide_clip_geom,
        line_tolerance=line_tolerance,
    )
    for layer, lines in final_linework.items():
        for line in lines:
            coords = _clean_coords(line.coords, close=False, collinear_tolerance=line_tolerance)
            for start, end in zip(coords, coords[1:]):
                if _same_point(start, end):
                    continue
                _add_dxf_line_entity(msp, start, end, layer)


def _finalize_linework_by_layer(lines_by_layer, cfg, guide_clip_geom=None, line_tolerance=0.0):
    priority = _linework_layer_priority(cfg, lines_by_layer)
    final = {}
    protected_linework = sg.GeometryCollection()
    guide_layers = {cfg["bed"]["name"], cfg["margin"]["name"]}

    for layer in priority:
        source_lines = lines_by_layer.get(layer, [])
        if layer in guide_layers and guide_clip_geom is not None and not guide_clip_geom.is_empty:
            clipped = []
            for line in source_lines:
                clipped.extend(_line_parts(line.difference(guide_clip_geom)))
            source_lines = clipped

        lines = _dedupe_linework(source_lines, line_tolerance)
        if not lines:
            final[layer] = []
            continue

        if not protected_linework.is_empty:
            trimmed = []
            for line in lines:
                remainder = line.difference(protected_linework)
                trimmed.extend(_line_parts(remainder))
            lines = _dedupe_linework(trimmed, line_tolerance)

        final[layer] = lines
        if lines:
            layer_union = unary_union(lines)
            protected_linework = (
                layer_union
                if protected_linework.is_empty
                else unary_union([protected_linework, layer_union])
            )

    return final


def _linework_layer_priority(cfg, lines_by_layer):
    preferred = [
        cfg["cut"]["name"],
        cfg["engrave"]["name"],
        cfg["mark"]["name"],
        cfg["margin"]["name"],
        cfg["bed"]["name"],
    ]
    extras = sorted(layer for layer in lines_by_layer if layer not in preferred)
    return preferred + extras


def _dedupe_linework(lines, line_tolerance=0.0):
    clean_lines = [
        part
        for line in lines
        for part in _line_parts(_clean_export_linework(line, line_tolerance))
        if part is not None and not part.is_empty and part.length > EPSILON
    ]
    if not clean_lines:
        return []

    merged = unary_union(clean_lines)
    try:
        merged = linemerge(merged)
    except ValueError:
        pass
    return [
        part
        for line in _line_parts(merged)
        for part in _line_parts(_clean_export_linework(line, line_tolerance))
        if part.length > EPSILON
    ]


def _add_dxf_line_entity(msp, start, end, layer):
    msp.add_line(
        start,
        end,
        dxfattribs=_line_dxf_attribs(layer),
    )


def _line_dxf_attribs(layer):
    return {
        "layer": layer,
        "color": 256,
        "linetype": "BYLAYER",
        "lineweight": -1,
    }


def _polygon_parts(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return []


def _line_parts(geom):
    if geom.is_empty:
        return []
    if geom.geom_type in ("LineString", "LinearRing"):
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        lines = []
        for part in geom.geoms:
            lines.extend(_line_parts(part))
        return lines
    return []


def _prepare_pieces_for_export(pieces, scale_factor, clean_tolerance, line_tolerance=0.0):
    return [
        _clean_piece_for_export(
            _scale_piece_for_export(piece, scale_factor),
            clean_tolerance,
            line_tolerance,
        )
        for piece in pieces
    ]


def _clean_piece_for_export(piece, clean_tolerance, line_tolerance=0.0):
    geom = _clean_export_geometry(piece["geom"], clean_tolerance)
    engravings = []
    for line in piece.get("engravings", []):
        source = _clean_export_linework(line, line_tolerance)
        if geom is not None and not geom.is_empty:
            source = source.intersection(geom)
        engravings.extend(_line_parts(_clean_export_linework(source, line_tolerance)))

    label_areas = []
    for area in piece.get("label_areas", []) or []:
        source = _clean_export_geometry(area, clean_tolerance)
        if geom is not None and not geom.is_empty and source is not None and not source.is_empty:
            source = source.intersection(geom)
        label_areas.extend(_polygon_parts(source))

    return {
        "id": piece.get("id"),
        "geom": geom,
        "engravings": engravings,
        "label_areas": label_areas,
        "elevation": piece.get("elevation"),
        "source_elevation": piece.get("source_elevation"),
    }


def _clean_export_linework(geom, tolerance):
    if geom is None or geom.is_empty:
        return sg.GeometryCollection()

    clean_parts = []
    for part in _line_parts(geom):
        source = part
        if tolerance > EPSILON:
            simplified = source.simplify(tolerance, preserve_topology=False)
            if not simplified.is_empty:
                source = simplified

        coords = list(source.coords)
        is_closed = len(coords) > 2 and (
            _same_point(coords[0], coords[-1]) or bool(getattr(source, "is_ring", False))
        )
        clean = _clean_coords(
            coords,
            close=is_closed,
            collinear_tolerance=tolerance,
        )
        if is_closed and len(clean) >= 3:
            clean.append(clean[0])
        if len(clean) < 2:
            continue
        clean_line = sg.LineString(clean)
        if clean_line.length > EPSILON:
            clean_parts.append(clean_line)

    if not clean_parts:
        return sg.GeometryCollection()
    if len(clean_parts) == 1:
        return clean_parts[0]
    return unary_union(clean_parts)


def _clean_export_geometry(geom, tolerance):
    if geom is None or geom.is_empty:
        return geom

    original = geom
    cleaned = geom.buffer(0)
    if cleaned.is_empty:
        return original

    if tolerance > EPSILON:
        closed = cleaned.buffer(tolerance, join_style=2).buffer(-tolerance, join_style=2)
        if not closed.is_empty:
            cleaned = closed.buffer(0)

        simplify_tolerance = min(tolerance / 5.0, 0.1)
        if simplify_tolerance > EPSILON:
            simplified = cleaned.simplify(simplify_tolerance, preserve_topology=True)
            if not simplified.is_empty:
                cleaned = simplified.buffer(0)

    if cleaned.is_empty:
        return original
    return cleaned


def _scale_piece_for_export(piece, factor):
    if abs(factor - 1.0) <= EPSILON:
        return piece

    return {
        "id": piece.get("id"),
        "geom": _scale_geometry(piece["geom"], factor),
        "engravings": [_scale_geometry(line, factor) for line in piece.get("engravings", [])],
        "label_areas": [
            _scale_geometry(area, factor)
            for area in piece.get("label_areas", []) or []
        ],
        "elevation": piece.get("elevation"),
        "source_elevation": piece.get("source_elevation"),
    }


def _translate_piece_for_export(piece, xoff, yoff):
    return {
        "id": piece.get("id"),
        "geom": _translate_geometry(piece["geom"], xoff, yoff),
        "engravings": [
            _translate_geometry(line, xoff, yoff)
            for line in piece.get("engravings", [])
        ],
        "label_areas": [
            _translate_geometry(area, xoff, yoff)
            for area in piece.get("label_areas", []) or []
        ],
        "elevation": piece.get("elevation"),
        "source_elevation": piece.get("source_elevation"),
    }


def _scale_geometry(geom, factor):
    if geom is None or geom.is_empty or abs(factor - 1.0) <= EPSILON:
        return geom
    return affinity_scale(geom, xfact=factor, yfact=factor, origin=(0.0, 0.0))


def _translate_geometry(geom, xoff, yoff):
    if geom is None or geom.is_empty:
        return geom
    return affinity_translate(geom, xoff=xoff, yoff=yoff)


def _combined_sheet_columns(sheet_count, bed_w, bed_h):
    if sheet_count <= 1:
        return 1
    ratio = max(bed_h, EPSILON) / max(bed_w, EPSILON)
    return max(1, int(math.ceil(math.sqrt(sheet_count * ratio))))


def _piece_label(piece):
    value = piece.get("source_elevation")
    if value is None:
        value = piece.get("elevation")
    return _fmt_label_number(value)


def _fmt_label_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""

    if abs(number - round(number)) <= 1e-6:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _label_box_for_piece(
    piece,
    label,
    bed_w,
    bed_h,
    obstacles,
    sheet_margin=0.0,
    text_size=TEXT_SIZE,
    sheet_origin=(0.0, 0.0),
):
    label_w, label_h = _text_dimensions(label, text_size=text_size)
    engraving_area = _piece_label_area(piece)
    if engraving_area is not None and not engraving_area.is_empty:
        inside_box = _label_box_inside_area(engraving_area, label_w, label_h)
        if inside_box is not None:
            return inside_box

    return _label_box_near_piece(
        piece["geom"],
        label,
        bed_w,
        bed_h,
        obstacles,
        sheet_margin=sheet_margin,
        text_size=text_size,
        sheet_origin=sheet_origin,
    )


def _piece_label_area(piece):
    explicit_areas = [
        area
        for area in piece.get("label_areas", []) or []
        if area is not None and not area.is_empty and area.area > EPSILON
    ]
    if explicit_areas:
        area = unary_union(explicit_areas)
        geom = piece.get("geom")
        if geom is not None and not geom.is_empty:
            area = area.intersection(geom)
        return area
    return _engraving_label_area(piece)


def _engraving_label_area(piece):
    polygons = []
    for engraving in piece.get("engravings", []):
        for part in _line_parts(engraving):
            coords = [(round(float(x), 6), round(float(y), 6)) for x, y in part.coords]
            if len(coords) < 4:
                continue
            if not (_same_point(coords[0], coords[-1]) or bool(getattr(part, "is_ring", False))):
                continue

            try:
                poly = sg.Polygon(coords)
            except (TypeError, ValueError):
                continue
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area <= EPSILON:
                continue
            polygons.extend(_polygon_parts(poly))

    if not polygons:
        return sg.GeometryCollection()

    area = unary_union(polygons)
    geom = piece.get("geom")
    if geom is not None and not geom.is_empty:
        area = area.intersection(geom)
    return area


def _label_box_inside_area(area, label_w, label_h):
    search_area = area.buffer(-TEXT_CLEARANCE)
    if search_area.is_empty:
        search_area = area

    for poly in sorted(_polygon_parts(search_area), key=lambda item: item.area, reverse=True):
        minx, miny, maxx, maxy = poly.bounds
        if maxx - minx < label_w or maxy - miny < label_h:
            continue

        candidates = []
        for point in (poly.representative_point(), poly.centroid):
            candidates.append((point.x - label_w / 2.0, point.y - label_h / 2.0))

        for fx in (0.5, 0.35, 0.65, 0.2, 0.8):
            for fy in (0.5, 0.35, 0.65, 0.2, 0.8):
                candidates.append((
                    minx + (maxx - minx) * fx - label_w / 2.0,
                    miny + (maxy - miny) * fy - label_h / 2.0,
                ))

        seen = set()
        for x, y in candidates:
            key = (round(x, 3), round(y, 3))
            if key in seen:
                continue
            seen.add(key)
            box = sg.box(x, y, x + label_w, y + label_h)
            if poly.covers(box):
                return (x, y, x + label_w, y + label_h)

    return None


def _label_box_near_piece(
    geom,
    label,
    bed_w,
    bed_h,
    obstacles,
    sheet_margin=0.0,
    text_size=TEXT_SIZE,
    sheet_origin=(0.0, 0.0),
):
    minx, miny, maxx, maxy = geom.bounds
    sheet_x, sheet_y = sheet_origin
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    label_w, label_h = _text_dimensions(label, text_size=text_size)
    gap = EXTERNAL_LABEL_GAP

    candidates = [
        (maxx + gap, miny - gap - label_h),
        (cx - label_w / 2.0, miny - gap - label_h),
        (maxx + gap, cy - label_h / 2.0),
        (maxx + gap, maxy + gap),
        (minx - gap - label_w, miny - gap - label_h),
        (cx - label_w / 2.0, maxy + gap),
        (minx - gap - label_w, cy - label_h / 2.0),
    ]

    for x, y in candidates:
        box = sg.box(x, y, x + label_w, y + label_h)
        if not _box_inside_sheet(box.bounds, bed_w, bed_h, sheet_margin, origin=sheet_origin):
            continue
        if any(box.distance(obstacle) < gap - EPSILON for obstacle in obstacles):
            continue
        return (x, y, x + label_w, y + label_h)

    point = geom.representative_point()
    min_x = sheet_x + sheet_margin
    min_y = sheet_y + sheet_margin
    max_x = max(min_x, sheet_x + bed_w - sheet_margin - label_w)
    max_y = max(min_y, sheet_y + bed_h - sheet_margin - label_h)
    x = max(min_x, min(max_x, point.x - label_w / 2.0))
    y = max(min_y, min(max_y, point.y - label_h / 2.0))
    return (x, y, x + label_w, y + label_h)


def _box_inside_sheet(bounds, bed_w, bed_h, sheet_margin=0.0, origin=(0.0, 0.0)):
    minx, miny, maxx, maxy = bounds
    sheet_x, sheet_y = origin
    return (
        minx >= sheet_x + sheet_margin - EPSILON
        and miny >= sheet_y + sheet_margin - EPSILON
        and maxx <= sheet_x + bed_w - sheet_margin + EPSILON
        and maxy <= sheet_y + bed_h - sheet_margin + EPSILON
    )


def _text_dimensions(label, text_size=TEXT_SIZE):
    width = sum(_glyph_advance(char) * text_size for char in str(label))
    return max(width, 0.1), text_size


def _text_stroke_lines(label, x, y, y_axis_down, text_size=TEXT_SIZE):
    cursor_x = x
    lines = []

    for char in str(label):
        for segment in _glyph_segments(char):
            coords = []
            for px, py in segment:
                sx = cursor_x + px * text_size
                sy = y + (1.0 - py) * text_size if y_axis_down else y + py * text_size
                coords.append((sx, sy))

            coords = _clean_coords(coords, close=False)
            if len(coords) >= 2:
                lines.append(coords)

        cursor_x += _glyph_advance(char) * text_size

    return lines


def _glyph_advance(char):
    if char == " ":
        return 0.45
    if char in ".:":
        return 0.35
    if char.lower() == "l":
        return 0.45
    return 0.75


def _glyph_segments(char):
    glyphs = {
        "0": [((0.1, 0.0), (0.65, 0.0), (0.65, 1.0), (0.1, 1.0), (0.1, 0.0))],
        "1": [((0.35, 0.0), (0.35, 1.0)), ((0.18, 0.82), (0.35, 1.0), (0.52, 1.0))],
        "2": [((0.1, 0.82), (0.25, 1.0), (0.58, 1.0), (0.68, 0.82), (0.1, 0.0), (0.68, 0.0))],
        "3": [((0.1, 1.0), (0.65, 1.0), (0.38, 0.52), (0.65, 0.52), (0.65, 0.0), (0.1, 0.0))],
        "4": [((0.62, 0.0), (0.62, 1.0)), ((0.1, 0.42), (0.68, 0.42)), ((0.1, 1.0), (0.1, 0.42))],
        "5": [((0.68, 1.0), (0.12, 1.0), (0.12, 0.55), (0.58, 0.55), (0.68, 0.42), (0.68, 0.0), (0.1, 0.0))],
        "6": [((0.65, 0.9), (0.52, 1.0), (0.15, 0.75), (0.1, 0.25), (0.28, 0.0), (0.65, 0.0), (0.65, 0.52), (0.12, 0.52))],
        "7": [((0.08, 1.0), (0.68, 1.0), (0.22, 0.0))],
        "8": [((0.15, 0.52), (0.1, 0.78), (0.25, 1.0), (0.55, 1.0), (0.68, 0.78), (0.58, 0.52), (0.15, 0.52), (0.1, 0.25), (0.28, 0.0), (0.55, 0.0), (0.68, 0.25), (0.58, 0.52))],
        "9": [((0.65, 0.48), (0.12, 0.48), (0.12, 1.0), (0.55, 1.0), (0.68, 0.75), (0.62, 0.25), (0.18, 0.0))],
        "E": [((0.65, 1.0), (0.1, 1.0), (0.1, 0.0), (0.65, 0.0)), ((0.1, 0.52), (0.55, 0.52))],
        "e": [((0.65, 0.18), (0.5, 0.0), (0.18, 0.0), (0.08, 0.2), (0.08, 0.55), (0.25, 0.75), (0.58, 0.75), (0.68, 0.55), (0.1, 0.42))],
        "l": [((0.18, 1.0), (0.18, 0.08), (0.38, 0.0))],
        "L": [((0.12, 1.0), (0.12, 0.0), (0.65, 0.0))],
        "v": [((0.08, 0.75), (0.35, 0.0), (0.65, 0.75))],
        ":": [((0.16, 0.68), (0.19, 0.68)), ((0.16, 0.25), (0.19, 0.25))],
        ".": [((0.16, 0.0), (0.19, 0.0))],
        "-": [((0.12, 0.48), (0.6, 0.48))],
        "_": [((0.08, 0.0), (0.65, 0.0))],
    }

    return glyphs.get(
        char,
        glyphs.get(char.upper(), [((0.1, 0.0), (0.65, 1.0)), ((0.65, 0.0), (0.1, 1.0))]),
    )


def _clean_coords(coords, close, collinear_tolerance=0.0):
    clean = []
    for x, y in coords:
        point = (round(float(x), 6), round(float(y), 6))
        if clean and _same_point(clean[-1], point):
            continue
        clean.append(point)

    if close and len(clean) > 1 and _same_point(clean[0], clean[-1]):
        clean.pop()

    return _remove_redundant_collinear_points(clean, close, collinear_tolerance)


def _remove_redundant_collinear_points(points, close, tolerance):
    min_points = 3 if close else 2
    if len(points) <= min_points:
        return points

    tolerance = max(float(tolerance or 0.0), EPSILON)
    cleaned = list(points)
    changed = True
    while changed and len(cleaned) > min_points:
        changed = False
        indices = range(len(cleaned)) if close else range(1, len(cleaned) - 1)
        for index in list(indices):
            if len(cleaned) <= min_points:
                break
            if not close and (index <= 0 or index >= len(cleaned) - 1):
                continue
            previous_point = cleaned[index - 1]
            current_point = cleaned[index]
            next_point = cleaned[(index + 1) % len(cleaned)]
            if _point_is_redundant_collinear(
                previous_point,
                current_point,
                next_point,
                tolerance,
            ):
                cleaned.pop(index)
                changed = True
                break
    return cleaned


def _point_is_redundant_collinear(previous_point, current_point, next_point, tolerance):
    ax, ay = previous_point
    bx, by = current_point
    cx, cy = next_point
    dx = cx - ax
    dy = cy - ay
    length = math.hypot(dx, dy)
    if length <= EPSILON:
        return True

    distance = abs(((bx - ax) * dy) - ((by - ay) * dx)) / length
    if distance > tolerance:
        return False

    dot_from_start = ((bx - ax) * dx) + ((by - ay) * dy)
    dot_from_end = ((bx - cx) * -dx) + ((by - cy) * -dy)
    projection_tolerance = tolerance * length
    return dot_from_start >= -projection_tolerance and dot_from_end >= -projection_tolerance


def _same_point(a, b):
    return abs(a[0] - b[0]) <= EPSILON and abs(a[1] - b[1]) <= EPSILON


def _fmt(value):
    return f"{float(value):.3f}".rstrip("0").rstrip(".")
