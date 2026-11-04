from std.math import cos, exp, sin, sqrt
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.kernels.ops import (
    attention_ranges,
    rope_apply,
    rope_table_factors,
)
from mojo_dllm.models.diffusion_gemma import KeyRange, key_range
from mojo_dllm.sys.mem import Bytes, F32Ptr, I32Ptr, Floats

comptime FIX = "tests/fixtures/"


# The branch's unified mask, one cell, transcribed from
# `llm_graph_input_attn_diffusion::set_input` (diffusion-gemma.cpp). Prompt rows
# are causal over the prompt, clipped to the window in sliding layers. Canvas rows
# see the whole canvas, every prompt token in a global layer, and in a sliding
# layer the prompt tokens from P - window + 1 on.
def _allowed(q: Int, k: Int, p: Int, window: Int, sliding: Bool) -> Bool:
    var q_canvas = q >= p
    var k_canvas = k >= p
    if q_canvas:
        if sliding:
            return k_canvas or k >= p - window + 1
        return True
    var ok = (not k_canvas) and k <= q
    if ok and sliding and q - k >= window:
        ok = False
    return ok


def test_key_range_is_the_branch_mask_cell_for_cell() raises:
    var cells = 0
    for p in [0, 1, 5, 9, 20]:
        for window in [1, 3, 8, 64]:
            for sliding in [False, True]:
                var n = p + 6
                for q in range(n):
                    var r = key_range(q, p, window, sliding, n)
                    for k in range(n):
                        var inside = k >= r.lo and k < r.hi
                        if inside != _allowed(q, k, p, window, sliding):
                            raise Error(
                                "mask differs at q="
                                + String(q)
                                + " k="
                                + String(k)
                                + " P="
                                + String(p)
                                + " window="
                                + String(window)
                                + " sliding="
                                + String(sliding)
                            )
                        cells += 1
    assert_true(cells > 5000, "the comparison covered " + String(cells))


def test_sliding_canvas_row_reaches_window_minus_one_prompt_tokens() raises:
    # P = 10, window 4: a canvas row sees prompt tokens 7, 8, 9 (the last
    # window - 1 = 3) and the canvas, not the 4 a prompt row would see.
    var r = key_range(12, 10, 4, True, 16)
    assert_equal(r.lo, 7)
    assert_equal(r.hi, 16)
    var g = key_range(12, 10, 4, False, 16)
    assert_equal(g.lo, 0)
    assert_equal(g.hi, 16)


def test_prompt_row_never_sees_the_canvas() raises:
    for sliding in [False, True]:
        var r = key_range(9, 10, 4, sliding, 16)
        assert_equal(r.hi, 10)
        assert_true(r.lo <= 9)
    var first = key_range(0, 10, 4, True, 16)
    assert_equal(first.lo, 0)
    assert_equal(first.hi, 1)
    var last = key_range(9, 10, 4, True, 16)
    assert_equal(last.lo, 6)
    assert_equal(last.hi, 10)


def test_rope_table_divides_the_angle_by_the_factor() raises:
    # head 8: four pairs. Factor 1e30 stops a pair from rotating; the others
    # rotate by p * theta^(-2i/8) / factor.
    var factors = Floats(4)
    factors[0] = 1.0
    factors[1] = 1e30
    factors[2] = 2.0
    factors[3] = 1e30
    var c = Floats(6 * 4)
    var s = Floats(6 * 4)
    rope_table_factors(6, 8, 1e6, 0, factors.ptr(), True, c.ptr(), s.ptr())
    var worst = Float64(0)
    for t in range(6):
        for i in range(4):
            var f = Float64(factors[i])
            var ang = Float64(t) * (Float64(1e6) ** (-2.0 * Float64(i) / 8.0)) / f
            worst = max(worst, abs(Float64(c[t * 4 + i]) - cos(ang)))
            worst = max(worst, abs(Float64(s[t * 4 + i]) - sin(ang)))
    assert_true(worst < 1e-6, "table error " + String(worst))
    for t in range(6):
        assert_equal(c[t * 4 + 1], 1.0)
        assert_true(abs(s[t * 4 + 1]) < 1e-20)
    # Pair 2 turns at half the rate it would without the factor.
    var plain = Floats(6 * 4)
    var plain_s = Floats(6 * 4)
    rope_table_factors(
        6, 8, 1e6, 0, factors.ptr(), False, plain.ptr(), plain_s.ptr()
    )
    assert_true(abs(plain_s[5 * 4 + 2] - s[5 * 4 + 2]) > 1e-4)


def test_neox_pairs_i_with_i_plus_half() raises:
    var c = Floats(2 * 4)
    var s = Floats(2 * 4)
    rope_table_factors(2, 8, 100.0, 0, c.ptr(), False, c.ptr(), s.ptr())
    var x = Floats(2 * 8)
    for i in range(16):
        x[i] = Float32(i % 8) + 1.0
    rope_apply(x.ptr(), 2, 1, 8, c.ptr(), s.ptr(), True, 1)
    for t in range(2):
        for i in range(4):
            var a = Float64(i) + 1.0
            var b = Float64(i + 4) + 1.0
            var ang = Float64(t) * (Float64(100.0) ** (-2.0 * Float64(i) / 8.0))
            var want_lo = a * cos(ang) - b * sin(ang)
            var want_hi = a * sin(ang) + b * cos(ang)
            assert_true(abs(Float64(x[t * 8 + i]) - want_lo) < 1e-5)
            assert_true(abs(Float64(x[t * 8 + i + 4]) - want_hi) < 1e-5)


def _attention_case(
    hd: Int, scale: Float32, n_q: Int, n_keys: Int
) raises -> Float64:
    var nh = 4
    var nkv = 2
    var q = Floats(n_q * nh * hd)
    var k = Floats(n_keys * nkv * hd)
    var v = Floats(n_keys * nkv * hd)
    for i in range(n_q * nh * hd):
        q[i] = Float32((i * 7919) % 97 - 48) / 40.0
    for i in range(n_keys * nkv * hd):
        k[i] = Float32((i * 104729) % 89 - 44) / 30.0
        v[i] = Float32((i * 1299709) % 83 - 41) / 20.0
    var lo = Bytes(n_q * 4)
    var hi = Bytes(n_q * 4)
    for i in range(n_q):
        # An empty-looking prefix, a single key, a middle window, everything.
        var a = [0, n_keys - 1, 2, 0][i % 4]
        var b = [3, n_keys, n_keys - 1, n_keys][i % 4]
        lo.i32().unsafe_store(i, Int32(a))
        hi.i32().unsafe_store(i, Int32(b))
    var o = Floats(n_q * nh * hd)
    attention_ranges(
        q.ptr(), k.ptr(), v.ptr(), o.ptr(), lo.i32(), hi.i32(),
        n_q, n_keys, nh, nkv, hd, scale, 3,
    )
    var worst = Float64(0)
    for i in range(n_q):
        var a = Int(lo.i32().unsafe_load(i))
        var b = Int(hi.i32().unsafe_load(i))
        for h in range(nh):
            var kvh = h // (nh // nkv)
            var sc = List[Float64]()
            var mx = Float64(-1e30)
            for j in range(a, b):
                var dot = Float64(0)
                for d in range(hd):
                    dot += Float64(q[(i * nh + h) * hd + d]) * Float64(
                        k[(j * nkv + kvh) * hd + d]
                    )
                sc.append(dot * Float64(scale))
                mx = max(mx, sc[len(sc) - 1])
            var z = Float64(0)
            for t in range(len(sc)):
                sc[t] = exp(sc[t] - mx)
                z += sc[t]
            for d in range(hd):
                var want = Float64(0)
                for j in range(a, b):
                    want += sc[j - a] / z * Float64(v[(j * nkv + kvh) * hd + d])
                var got = Float64(o[(i * nh + h) * hd + d])
                worst = max(worst, abs(got - want))
    return worst


def test_attention_ranges_match_a_scalar_reference() raises:
    for hd in [16, 64]:
        var worst = _attention_case(hd, 1.0, 9, 12)
        assert_true(worst < 2e-5, "hd " + String(hd) + " error " + String(worst))


def test_attention_scale_is_the_argument_not_one_over_root_d() raises:
    var worst = _attention_case(64, 0.5, 9, 12)
    assert_true(worst < 2e-5, "scale 0.5 error " + String(worst))
    var worst_root = _attention_case(64, 0.125, 9, 12)
    assert_true(worst_root < 2e-5, "scale 1/8 error " + String(worst_root))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
