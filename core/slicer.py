import math

import networkx as nx
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import linemerge, snap, unary_union
from shapely.affinity import scale

HOLLOW_SUPPORT_RULE_VERSION = "visible_buffer_clip_clean_restore_v1"
SOLID_VISIBILITY_RULE_VERSION = "solid_visibility_v1"
OFFSET_SIMPLIFY_RATIO = 0.02


def generate_rings(
    topology_tree,
    engrave_offset=1.0,
    scale_factor=1.0,
    direction_changes=None,
    slicing_mode="solid",
    boundary_layer="boundary",
    hollow_glue_margin=0.0,
):
    """Generate slice geometries for each node.

    - If slicing_mode is "solid", each sheet is boundary minus the level polygon (unless descending).
    - If slicing_mode is "hollow", each sheet is a hollow ring (parent minus children).
    - Engraving lines are created by buffering each child polygon.
    - All geometries are scaled by ``scale_factor`` before returning.
    """
    if direction_changes is None:
        direction_changes = set()

    dxf_engrave_offset = engrave_offset
    slices = []

    # 1. Encontrar o polígono de limite (boundary_geom)
    boundary_geom = None
    for node in topology_tree.nodes():
        data = topology_tree.nodes[node]['data']
        layer_name = data.get('layer', '').lower()
        if boundary_layer is not None and layer_name == boundary_layer.lower():
            boundary_geom = data['geometry']
            break
            
    # Fallback se não encontrar o layer boundary: usar o maior polígono (raiz do grafo)
    if boundary_geom is None:
        roots = [n for n, d in topology_tree.in_degree() if d == 0]
        if roots:
            roots.sort(key=lambda n: topology_tree.nodes[n]['data']['geometry'].area, reverse=True)
            boundary_geom = topology_tree.nodes[roots[0]]['data']['geometry']

    # Collect elevation info and sort bottom-up
    node_info = []
    for node in topology_tree.nodes():
        data = topology_tree.nodes[node]['data']
        elev = data.get('elevation')
        if elev is None:
            continue
        node_info.append((node, elev, data['geometry'], data.get('is_descending', False)))
    node_info.sort(key=lambda x: x[1])  # lowest elevation first

    for node_id, elev, geom, is_descending in node_info:
        visible_geom = None
        support_geom = Polygon()
        hidden_base_geom = Polygon()
        if slicing_mode == "solid" and boundary_geom is not None and geom.equals(boundary_geom):
            # A chapa base do boundary é sempre totalmente sólida no modo sólido
            slice_geom = boundary_geom
        else:
            if slicing_mode == "solid":
                if is_descending:
                    # Nível descendente: não subtrai, fica sólido
                    slice_geom = boundary_geom
                else:
                    # Nível ascendente: subtrai o polígono da curva ao limite (boundary)
                    if boundary_geom is not None:
                        slice_geom = boundary_geom.difference(geom)
                    else:
                        slice_geom = geom
            else:
                # "hollow" mode (Option B)
                children = list(topology_tree.successors(node_id))
                children_geoms = [topology_tree.nodes[c]['data']['geometry'] for c in children]
                glue_limit_geom = boundary_geom
                if children_geoms:
                    children_union = unary_union(children_geoms)
                    if node_id in direction_changes:
                        slice_geom, visible_geom, support_geom, hidden_base_geom = _hollow_layer_geometry(
                            children_union,
                            geom,
                            glue_limit_geom,
                            hollow_glue_margin,
                        )
                    else:
                        slice_geom, visible_geom, support_geom, hidden_base_geom = _hollow_layer_geometry(
                            geom,
                            children_union,
                            glue_limit_geom,
                            hollow_glue_margin,
                        )
                else:
                    slice_geom, visible_geom, support_geom, hidden_base_geom = _hollow_layer_geometry(
                        geom,
                        None,
                        glue_limit_geom,
                        hollow_glue_margin,
                    )

        if visible_geom is None:
            visible_geom = slice_geom

        # Create engraving lines from each child polygon
        children = list(topology_tree.successors(node_id))
        children_geoms = [topology_tree.nodes[c]['data']['geometry'] for c in children]
        engraving_lines = []
        for child_geom in children_geoms:
            engrave = child_geom.buffer(-dxf_engrave_offset)
            if not engrave.is_empty:
                if isinstance(engrave, MultiPolygon):
                    for poly in engrave.geoms:
                        engraving_lines.append(poly.exterior)
                else:
                    engraving_lines.append(engrave.exterior)

        # Apply scaling
        if not slice_geom.is_empty:
            slice_geom = scale(slice_geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        if visible_geom is not None and not visible_geom.is_empty:
            visible_geom = scale(visible_geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        if support_geom is not None and not support_geom.is_empty:
            support_geom = scale(support_geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        if hidden_base_geom is not None and not hidden_base_geom.is_empty:
            hidden_base_geom = scale(hidden_base_geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        scaled_engravings = []
        for line in engraving_lines:
            if not line.is_empty:
                scaled_engravings.append(scale(line, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)))

        slices.append({
            "id": node_id,
            "elevation": elev,
            "slice_geom": slice_geom,
            "visible_geom": visible_geom,
            "support_geom": support_geom,
            "hidden_base_geom": hidden_base_geom,
            "support_rule_version": HOLLOW_SUPPORT_RULE_VERSION if slicing_mode == "hollow" else None,
            "support_glue_margin": float(hollow_glue_margin or 0.0) if slicing_mode == "hollow" else 0.0,
            "engraving_lines": scaled_engravings,
        })

    if slicing_mode == "solid":
        return _annotate_solid_slice_visibility(slices)
    return slices


def _annotate_solid_slice_visibility(slices):
    if not slices:
        return []

    physical_geoms = [
        _solid_physical_geometry_from_metadata(item)
        for item in slices
    ]

    annotated = []
    for index, current in enumerate(slices):
        current_geom = physical_geoms[index]
        if current_geom is None or current_geom.is_empty:
            continue

        next_geom = physical_geoms[index + 1] if index < len(physical_geoms) - 1 else Polygon()
        if next_geom is not None and not next_geom.is_empty:
            hidden_base_geom = _clean_polygonal_geometry(current_geom.intersection(next_geom))
            visible_geom = _clean_polygonal_geometry(current_geom.difference(next_geom))
        else:
            hidden_base_geom = Polygon()
            visible_geom = current_geom

        item = dict(current)
        item["slice_geom"] = current_geom
        item["visible_geom"] = visible_geom
        item["hidden_base_geom"] = hidden_base_geom
        item["support_geom"] = hidden_base_geom
        item["support_rule_version"] = SOLID_VISIBILITY_RULE_VERSION
        item["support_glue_margin"] = 0.0
        annotated.append(item)

    return annotated


def _solid_physical_geometry_from_metadata(item):
    visible_geom = item.get("visible_geom")
    hidden_base_geom = item.get("hidden_base_geom")
    if (
        visible_geom is not None
        and not visible_geom.is_empty
        and hidden_base_geom is not None
        and not hidden_base_geom.is_empty
    ):
        return _clean_polygonal_geometry(unary_union([visible_geom, hidden_base_geom]))

    return _clean_polygonal_geometry(item.get("slice_geom"))


def _clean_polygonal_geometry(geom):
    geom = _polygonal_geometry(geom)
    if geom.is_empty:
        return Polygon()
    if not geom.is_valid:
        geom = geom.buffer(0)
    return _polygonal_geometry(geom)


def _hollow_layer_geometry(base_geom, cut_geom, limit_geom, glue_margin):
    if limit_geom is not None and not limit_geom.is_empty:
        base_geom = base_geom.intersection(limit_geom)

    if cut_geom is None or cut_geom.is_empty:
        visible_geom = base_geom
        hidden_geom = Polygon()
    else:
        visible_geom = base_geom.difference(cut_geom)
        hidden_geom = base_geom.intersection(cut_geom)

    if glue_margin is None or glue_margin <= 0:
        return visible_geom, visible_geom, Polygon(), hidden_geom

    support_geom = _hidden_support_geometry(visible_geom, hidden_geom, glue_margin)
    if support_geom.is_empty:
        return visible_geom, visible_geom, support_geom, hidden_geom

    return unary_union([visible_geom, support_geom]), visible_geom, support_geom, hidden_geom


def _hidden_support_geometry(visible_geom, hidden_geom, glue_margin):
    if hidden_geom is None or hidden_geom.is_empty or glue_margin is None or glue_margin <= 0:
        return Polygon()
    if visible_geom is None or visible_geom.is_empty:
        return Polygon()

    support_geom = visible_geom.buffer(
        float(glue_margin),
        join_style=2,
        mitre_limit=2.0,
    ).intersection(hidden_geom)

    support_geom = _clean_support_geometry(support_geom, visible_geom, glue_margin)
    return support_geom if not support_geom.is_empty else Polygon()


def _shared_visible_hidden_contact_line(visible_geom, hidden_geom, glue_margin):
    if visible_geom is None or visible_geom.is_empty or hidden_geom is None or hidden_geom.is_empty:
        return LineString()

    shared = visible_geom.boundary.intersection(hidden_geom.boundary)
    if shared.is_empty:
        shared = visible_geom.boundary.intersection(hidden_geom)
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
    return sum(line.length for line in _linear_parts(contact))


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


def _offset_contact_line_to_hidden_side(contact_line, hidden_geom, glue_margin):
    if _contact_margin_covers_base(contact_line, hidden_geom, glue_margin):
        return hidden_geom

    supports = []
    for line in _linear_parts(contact_line):
        support = _best_hidden_side_single_offset(line, hidden_geom, glue_margin)
        if support is not None and not support.is_empty:
            supports.append(support)

    if not supports:
        return Polygon()

    support = unary_union(supports).intersection(hidden_geom)
    return _polygonal_geometry(support)


def _contact_margin_covers_base(contact_line, hidden_geom, glue_margin):
    if contact_line is None or contact_line.is_empty or hidden_geom is None or hidden_geom.is_empty:
        return False
    coverage = contact_line.buffer(
        float(glue_margin),
        cap_style=2,
        join_style=2,
        mitre_limit=2.0,
    )
    if coverage.is_empty:
        return False
    remainder = hidden_geom.difference(coverage)
    return remainder.is_empty or remainder.area <= max(1e-6, hidden_geom.area * 1e-6)


def _best_hidden_side_single_offset(line, hidden_geom, glue_margin):
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
        support = strip.intersection(hidden_geom)
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


def _polygonal_geometry(geom):
    parts = _polygon_parts(geom)
    if not parts:
        return Polygon()
    return unary_union(parts)


def _polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts = []
        for child in geom.geoms:
            parts.extend(_polygon_parts(child))
        return parts
    return []


def _simplify_contact_line(geom, glue_margin):
    if geom is None or geom.is_empty or glue_margin is None or glue_margin <= 0:
        return LineString()

    tolerance = min(float(glue_margin) * OFFSET_SIMPLIFY_RATIO, 0.25)
    min_length = max(tolerance, 1e-6)
    source_lines = [
        line
        for line in _linear_parts(geom)
        if line.length > min_length
    ]
    if not source_lines:
        return LineString()

    merged = unary_union(source_lines)
    merged = snap(merged, merged, max(tolerance, 1e-6))
    merged = unary_union(_linear_parts(merged))
    try:
        merged = linemerge(merged)
    except ValueError:
        pass

    simplified_lines = []
    for line in _linear_parts(merged):
        if line.length <= min_length:
            continue
        simplified = line.simplify(tolerance, preserve_topology=False) if tolerance > 0 else line
        if simplified.is_empty:
            simplified = line
        simplified_lines.extend(
            part for part in _linear_parts(simplified) if part.length > min_length
        )

    if not simplified_lines:
        return LineString()

    merged = unary_union(simplified_lines)
    merged = snap(merged, merged, max(tolerance, 1e-6))
    merged = unary_union(_linear_parts(merged))
    try:
        merged = linemerge(merged)
    except ValueError:
        pass
    return merged


def _linear_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type in ("LineString", "LinearRing"):
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts = []
        for child in geom.geoms:
            parts.extend(_linear_parts(child))
        return parts
    return []
