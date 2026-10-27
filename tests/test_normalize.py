import numpy as np

from tdlsv.preprocessing.normalize import MIDDLE_MCP, WRIST, normalize


def _synthetic_hand(wrist_xy, mcp_xy):
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[WRIST] = [wrist_xy[0], wrist_xy[1], 0.0]
    pts[MIDDLE_MCP] = [mcp_xy[0], mcp_xy[1], 0.0]
    for i in range(21):
        if i not in (WRIST, MIDDLE_MCP):
            pts[i] = [0.5 + 0.01 * i, 0.8 - 0.01 * i, 0.001 * i]
    return pts


_FACE = np.array([0.5, 0.3], dtype=np.float32)


def _hand_part(out: np.ndarray) -> np.ndarray:
    return out[:63].reshape(21, 3)


def _face_part(out: np.ndarray) -> np.ndarray:
    return out[63:]


def test_wrist_is_translated_to_origin():
    pts = _synthetic_hand(wrist_xy=(0.5, 0.8), mcp_xy=(0.6, 0.5))
    out = _hand_part(normalize(pts, "Right", _FACE))
    assert np.allclose(out[WRIST], [0.0, 0.0, 0.0], atol=1e-5)


def test_mcp_points_up_after_rotation():
    pts = _synthetic_hand(wrist_xy=(0.5, 0.8), mcp_xy=(0.6, 0.5))
    out = _hand_part(normalize(pts, "Right", _FACE))
    mcp = out[MIDDLE_MCP]
    assert np.allclose(mcp[:2], [0.0, -1.0], atol=1e-5)


def test_scale_is_normalized():
    pts = _synthetic_hand(wrist_xy=(0.5, 0.8), mcp_xy=(0.6, 0.5))
    out = _hand_part(normalize(pts, "Right", _FACE))
    assert np.isclose(np.linalg.norm(out[MIDDLE_MCP][:2]), 1.0, atol=1e-5)


def test_left_hand_is_mirrored_to_match_right():
    pts = _synthetic_hand(wrist_xy=(0.5, 0.8), mcp_xy=(0.6, 0.5))
    out_right = normalize(pts, "Right", _FACE)
    out_left = normalize(pts, "Left", _FACE)
    hand_right, hand_left = _hand_part(out_right), _hand_part(out_left)
    assert np.allclose(hand_left[:, 0], -hand_right[:, 0], atol=1e-5)
    assert np.allclose(hand_left[:, 1:], hand_right[:, 1:], atol=1e-5)
    assert np.isclose(_face_part(out_left)[0], -_face_part(out_right)[0], atol=1e-5)
    assert np.isclose(_face_part(out_left)[1], _face_part(out_right)[1], atol=1e-5)


def test_output_shape_and_dtype():
    pts = _synthetic_hand(wrist_xy=(0.5, 0.8), mcp_xy=(0.6, 0.5))
    out = normalize(pts, "Right", _FACE)
    assert out.shape == (65,)
    assert out.dtype == np.float32


def test_degenerate_scale_does_not_crash():
    pts = _synthetic_hand(wrist_xy=(0.5, 0.8), mcp_xy=(0.5, 0.8))
    out = normalize(pts, "Right", _FACE)
    assert np.isfinite(out).all()


def test_face_point_above_eye_vs_in_front_are_distinguishable():
    """Reproduce la diferencia real entre C y E: misma forma de mano, la
    cara/ojo en distinta posición relativa a la muñeca."""
    pts = _synthetic_hand(wrist_xy=(0.5, 0.8), mcp_xy=(0.6, 0.5))
    face_like_e = np.array([0.6, 0.55], dtype=np.float32)  # cerca del nudillo (a la altura del ojo)
    face_like_c = np.array([0.6, 0.75], dtype=np.float32)  # más lejos (más abajo/adelante)
    out_e = _face_part(normalize(pts, "Right", face_like_e))
    out_c = _face_part(normalize(pts, "Right", face_like_c))
    assert not np.allclose(out_e, out_c, atol=1e-3)
