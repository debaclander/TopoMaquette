import ezdxf


def create_test_dxf(filename="test_topography.dxf"):
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    doc.layers.new(name="boundary", dxfattribs={"color": 1})
    doc.layers.new(name="topografia", dxfattribs={"color": 2})
    doc.layers.new(name="3D_PONTOS", dxfattribs={"color": 5})

    # Nested topographic contours. The shape is intentionally simple but reads
    # as terrain instead of a building/house symbol.
    shapes = [
        [(24, 55), (45, 38), (74, 31), (103, 34), (132, 48), (150, 77), (143, 106), (119, 133), (86, 144), (55, 134), (34, 109), (20, 82)],
        [(42, 65), (58, 52), (80, 46), (104, 50), (125, 63), (134, 84), (127, 105), (109, 122), (84, 128), (62, 119), (48, 100), (38, 80)],
        [(60, 74), (75, 62), (95, 61), (113, 72), (120, 88), (114, 104), (98, 115), (78, 114), (64, 101), (56, 86)],
        [(75, 82), (88, 75), (101, 79), (109, 91), (103, 101), (90, 106), (78, 99), (72, 90)],
        [(86, 88), (94, 87), (99, 93), (94, 99), (86, 98), (82, 92)],
    ]

    # The boundary matches the lowest layer so the generated preview starts
    # from the terrain footprint instead of a separate square sheet.
    msp.add_lwpolyline(
        shapes[0],
        close=True,
        dxfattribs={"layer": "boundary"},
    )

    for points in shapes[1:]:
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "topografia"})

    # 3D points placed on contour boundaries for automatic elevation tests.
    point_data = [
        (24, 55, 0.0),
        (103, 34, 0.0),
        (86, 144, 0.0),
        (42, 65, 0.5),
        (125, 63, 0.5),
        (84, 128, 0.5),
        (60, 74, 1.0),
        (120, 88, 1.0),
        (78, 114, 1.0),
        (75, 82, 1.5),
        (103, 101, 1.5),
        (86, 88, 2.0),
        (94, 99, 2.0),
    ]
    for x, y, z in point_data:
        msp.add_point((x, y, z), dxfattribs={"layer": "3D_PONTOS"})

    doc.saveas(filename)
    print(f"DXF de teste gerado em {filename}")
    print("  Layer 'boundary': 1 poligono igual a camada mais baixa")
    print("  Layer 'topografia': 4 poligonos fechados")
    print("  Layer '3D_PONTOS': 13 pontos 3D")


if __name__ == "__main__":
    create_test_dxf()
