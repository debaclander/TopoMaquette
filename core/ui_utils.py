import math


def normalize_override_map(raw_map):
    normalized = {}
    if not isinstance(raw_map, dict):
        return normalized

    for contour_id, override in raw_map.items():
        key = str(contour_id).strip()
        if key == "" or not isinstance(override, dict):
            continue

        cleaned = {}
        for field, value in override.items():
            if field == "keep":
                cleaned[field] = value
                continue
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            cleaned[field] = value

        if cleaned:
            normalized[key] = cleaned

    return normalized


def merge_override_maps(base_map, priority_map):
    merged = {}

    for key, value in normalize_override_map(base_map).items():
        merged[key] = dict(value)

    for key, value in normalize_override_map(priority_map).items():
        row = dict(merged.get(key, {}))
        row.update(value)
        merged[key] = row

    return merged


def resolve_effective_overrides(points_map, summary_map, last_source):
    points_clean = normalize_override_map(points_map)
    summary_clean = normalize_override_map(summary_map)
    if last_source == "summary":
        return merge_override_maps(points_clean, summary_clean)
    return merge_override_maps(summary_clean, points_clean)


def resolve_boundary_and_contours(polygons, boundary_layer_name):
    if not polygons:
        return None, [], [], None

    if boundary_layer_name is not None:
        boundary_polygons = [p for p in polygons if p["layer"] == boundary_layer_name]
    else:
        boundary_polygons = [p for p in polygons if p["layer"].lower() == "boundary"]

    if boundary_polygons:
        outer_boundary = max(boundary_polygons, key=lambda p: p["geometry"].area)
        boundary_geom = outer_boundary["geometry"]
    else:
        outer_boundary = None
        boundary_geom = max((p["geometry"] for p in polygons), key=lambda geom: geom.area)

    if boundary_polygons:
        contour_polygons = [p for p in polygons if p not in boundary_polygons]
    else:
        contour_polygons = list(polygons)

    if outer_boundary is not None:
        contour_polygons.extend(
            p for p in boundary_polygons
            if p["id"] != outer_boundary["id"]
        )

    return boundary_geom, contour_polygons, boundary_polygons, outer_boundary


def parse_float_or_none(value, accept_comma=True):
    if value is None:
        return None

    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan"}:
        return None

    if accept_comma:
        text = text.replace(",", ".")

    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed):
        return None

    return parsed


def parse_int_or_none(value, accept_comma=True):
    parsed = parse_float_or_none(value, accept_comma=accept_comma)
    if parsed is None:
        return None

    rounded = round(parsed)
    if abs(parsed - rounded) > 1e-9:
        return None
    return int(rounded)


def parse_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0

    return str(value).strip().lower() not in {"0", "false", "f", "no", "n", "nao"}


def parse_curve_ids(value):
    if value is None:
        return []

    text = str(value).strip()
    if text == "":
        return []

    result = []
    for part in text.split(","):
        token = part.strip()
        if token == "":
            continue

        number = parse_int_or_none(token, accept_comma=False)
        if number is None:
            return None
        result.append(number)
    return result


def contour_id_sort_key(value):
    text = str(value).strip()
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (1, text.lower())
