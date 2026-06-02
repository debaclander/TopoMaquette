import re
from collections import Counter

import ezdxf
from shapely.geometry import Point, Polygon
from shapely.validation import explain_validity


def build_dxf_diagnostics(
    file_path,
    overlap_area_tolerance=1e-6,
    duplicate_area_tolerance=1e-4,
    max_overlap_examples=500,
):
    """
    Analyze closed DXF polylines and return quality diagnostics.

    Includes:
    - raw closed-polyline counts by layer
    - valid/invalid polygon split
    - invalid geometry reasons (including self-intersections)
    - overlap/nested/duplicate relationships with locations
    """
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    closed_by_layer = Counter()
    valid_by_layer = Counter()
    invalid_by_layer = Counter()
    invalid_shapes = []
    valid_shapes = []

    closed_total = 0
    shape_id = 0

    for entity in msp.query("LWPOLYLINE POLYLINE"):
        points = _extract_entity_points(entity)
        if len(points) < 3:
            continue

        closed_type = _closed_type(entity, points)
        if closed_type is None:
            continue

        layer = entity.dxf.layer
        closed_total += 1
        closed_by_layer[layer] += 1
        shape_id += 1

        if points[0] != points[-1]:
            points = points + [points[0]]

        polygon = _build_polygon(points)
        if polygon is not None and polygon.is_valid and not polygon.is_empty and polygon.area > 0:
            valid_by_layer[layer] += 1
            valid_shapes.append(
                {
                    "shape_id": shape_id,
                    "layer": layer,
                    "geometry": polygon,
                    "closed_type": closed_type,
                }
            )
            continue

        invalid_by_layer[layer] += 1
        reason = explain_validity(polygon) if polygon is not None else "Invalid polygon geometry"
        location = _invalid_location(reason, polygon, points)
        invalid_shapes.append(
            {
                "shape_id": shape_id,
                "layer": layer,
                "reason": reason,
                "closed_type": closed_type,
                "x": round(location.x, 6),
                "y": round(location.y, 6),
                "point_count": len(points) - 1,
                "points": _sample_polyline_points(points),
            }
        )

    overlaps, overlap_counters = _find_overlaps(
        valid_shapes,
        overlap_area_tolerance=overlap_area_tolerance,
        duplicate_area_tolerance=duplicate_area_tolerance,
        max_examples=max_overlap_examples,
    )

    layers = []
    all_layers = sorted(set(closed_by_layer) | set(valid_by_layer) | set(invalid_by_layer))
    for layer in all_layers:
        layers.append(
            {
                "layer": layer,
                "closed_total": int(closed_by_layer[layer]),
                "valid_total": int(valid_by_layer[layer]),
                "invalid_total": int(invalid_by_layer[layer]),
            }
        )

    return {
        "closed_total": int(closed_total),
        "valid_total": int(sum(valid_by_layer.values())),
        "invalid_total": int(sum(invalid_by_layer.values())),
        "layers": layers,
        "invalid_shapes": invalid_shapes,
        "overlap_pairs": overlaps,
        "overlap_summary": {key: int(value) for key, value in overlap_counters.items()},
    }


def _extract_entity_points(entity):
    try:
        if entity.dxftype() == "LWPOLYLINE":
            return [(point[0], point[1]) for point in entity.get_points()]
        return [(vertex.dxf.location.x, vertex.dxf.location.y) for vertex in entity.vertices]
    except Exception:
        return []


def _closed_type(entity, points):
    if bool(getattr(entity, "is_closed", False)):
        return "closed_flag"

    dx = abs(points[0][0] - points[-1][0])
    dy = abs(points[0][1] - points[-1][1])
    if dx < 1e-6 and dy < 1e-6:
        return "closed_coordinates"
    return None


def _build_polygon(points):
    try:
        return Polygon(points)
    except Exception:
        return None


def _sample_polyline_points(points, max_points=600):
    if len(points) <= max_points:
        return points

    step = max(int(len(points) / max_points), 1)
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _invalid_location(reason, polygon, points):
    match = re.search(r"\[([\-0-9.eE]+)\s+([\-0-9.eE]+)\]", str(reason))
    if match:
        try:
            return Point(float(match.group(1)), float(match.group(2)))
        except ValueError:
            pass

    if polygon is not None and not polygon.is_empty:
        point = polygon.representative_point()
        if point is not None and not point.is_empty:
            return point

    if points:
        return Point(points[0][0], points[0][1])
    return Point(0.0, 0.0)


def _find_overlaps(valid_shapes, overlap_area_tolerance, duplicate_area_tolerance, max_examples):
    overlaps = []
    counters = Counter()

    for index_a in range(len(valid_shapes)):
        item_a = valid_shapes[index_a]
        geom_a = item_a["geometry"]
        minx_a, miny_a, maxx_a, maxy_a = geom_a.bounds

        for index_b in range(index_a + 1, len(valid_shapes)):
            item_b = valid_shapes[index_b]
            geom_b = item_b["geometry"]

            minx_b, miny_b, maxx_b, maxy_b = geom_b.bounds
            if (
                maxx_a < minx_b
                or maxx_b < minx_a
                or maxy_a < miny_b
                or maxy_b < miny_a
            ):
                continue

            intersection = geom_a.intersection(geom_b)
            if intersection.is_empty:
                continue

            intersection_area = float(intersection.area)
            if intersection_area <= overlap_area_tolerance:
                continue

            relation_type = "overlap"
            sym_diff_area = float(geom_a.symmetric_difference(geom_b).area)
            if sym_diff_area <= duplicate_area_tolerance:
                relation_type = "duplicate"
            elif geom_a.within(geom_b) or geom_b.within(geom_a):
                relation_type = "nested"

            counters[relation_type] += 1

            if len(overlaps) >= max_examples:
                continue

            center = intersection.representative_point()
            overlaps.append(
                {
                    "shape_a": item_a["shape_id"],
                    "shape_b": item_b["shape_id"],
                    "layer_a": item_a["layer"],
                    "layer_b": item_b["layer"],
                    "relation": relation_type,
                    "intersection_area": round(intersection_area, 6),
                    "x": round(center.x, 6),
                    "y": round(center.y, 6),
                }
            )

    return overlaps, counters
