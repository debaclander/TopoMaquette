import io
from pathlib import Path

import ezdxf
from shapely.geometry import Point, Polygon


def _read_dxf_document(file_path_or_stream):
    """Read a DXF from a path or from an uploaded stream."""
    try:
        if isinstance(file_path_or_stream, (str, Path)):
            return ezdxf.readfile(str(file_path_or_stream))

        if hasattr(file_path_or_stream, "getvalue"):
            raw_content = file_path_or_stream.getvalue()
        else:
            raw_content = file_path_or_stream.read()

        if isinstance(raw_content, bytes):
            raw_content = raw_content.decode("utf-8", errors="ignore")

        return ezdxf.read(io.StringIO(raw_content))
    except IOError:
        print("Nao foi possivel abrir o ficheiro DXF.")
    except ezdxf.DXFStructureError:
        print("Ficheiro DXF invalido ou corrompido.")
    return None


def extract_polygons_from_dxf(file_path_or_stream):
    """
    Le um ficheiro DXF e extrai todas as polilinhas fechadas como Shapely Polygons.
    Retorna uma lista de dicionarios com id, layer e a geometria.
    """
    doc = _read_dxf_document(file_path_or_stream)
    if doc is None:
        return []

    msp = doc.modelspace()
    polygons = []

    poly_id = 0
    for entity in msp.query("LWPOLYLINE POLYLINE"):
        try:
            if entity.dxftype() == "LWPOLYLINE":
                points = [(p[0], p[1]) for p in entity.get_points()]
            else:
                points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
        except AttributeError:
            continue

        if len(points) < 3:
            continue

        is_closed = entity.is_closed
        if not is_closed:
            dx = abs(points[0][0] - points[-1][0])
            dy = abs(points[0][1] - points[-1][1])
            is_closed = dx < 1e-6 and dy < 1e-6

        if not is_closed:
            continue

        if points[0] != points[-1]:
            points.append(points[0])

        try:
            poly = Polygon(points)
            if poly.is_valid and not poly.is_empty:
                polygons.append(
                    {
                        "id": poly_id,
                        "layer": entity.dxf.layer,
                        "geometry": poly,
                        "elevation": 0.0,
                    }
                )
                poly_id += 1
        except Exception:
            # Ignore malformed entities and continue with the remaining data.
            pass

    return polygons


def extract_3d_points_from_dxf(file_path_or_stream, layer_name="3D_PONTOS"):
    """
    Le pontos 3D de uma layer DXF e devolve dicionarios com x, y, z e geometry.
    Se layer_name for None, devolve pontos de todas as layers.
    """
    doc = _read_dxf_document(file_path_or_stream)
    if doc is None:
        return []

    target_layer = layer_name.lower() if layer_name else None
    points = []

    for entity in doc.modelspace().query("POINT"):
        if target_layer is not None and entity.dxf.layer.lower() != target_layer:
            continue

        location = entity.dxf.location
        points.append(
            {
                "x": location.x,
                "y": location.y,
                "z": location.z,
                "layer": entity.dxf.layer,
                "geometry": Point(location.x, location.y),
            }
        )

    return points
