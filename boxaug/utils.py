def IOU(a_wh, b_wh):
    """
    Intersection over Union

    Args:
        a_wh: (width, height) of box A
        b_wh: (width, height) of box B

    Returns float.
    """
    aw, ah = a_wh
    bw, bh = b_wh

    I = min(aw, bw) * min(ah, bh)

    area_a = aw * ah
    area_b = bw * bh

    U = area_a + area_b - I

    return I / U
