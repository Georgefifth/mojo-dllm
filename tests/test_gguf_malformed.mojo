from std.testing import assert_equal, assert_raises, TestSuite

from mojo_dllm.formats.gguf import GGUFFile, GGML_Q4_K, Reader, ggml_row_bytes
from mojo_dllm.quant.kquants import dequant_row
from mojo_dllm.kernels.packed import PackedMatrix
from mojo_dllm.sys.mem import Bytes, Floats

comptime FIX = "tests/fixtures/malformed/"


def test_base_fixture_is_valid() raises:
    var g = GGUFFile(FIX + "base.gguf")
    assert_equal(g.n_tensors(), 2)
    assert_equal(g.tensor("f32").n_elements(), 8)
    assert_equal(g.tensor("q4k").n_elements(), 256)
    assert_equal(g.tensor("q4k").ggml_type, GGML_Q4_K)
    assert_equal(len(g.get_int_array("test.array")), 3)
    assert_equal(g.alignment, 32)
    assert_equal(g.tensor("f32").data_offset % 32, 0)


def test_rejects_zero_alignment() raises:
    with assert_raises(contains="general.alignment has invalid value 0"):
        _ = GGUFFile(FIX + "alignment-zero.gguf")


def test_rejects_nonpower_alignment() raises:
    with assert_raises(contains="general.alignment has invalid value 24"):
        _ = GGUFFile(FIX + "alignment-nonpower.gguf")


def test_rejects_alignment_wrong_type() raises:
    with assert_raises(
        contains="general.alignment has type 0, expected UINT32"
    ):
        _ = GGUFFile(FIX + "alignment-type.gguf")


def test_rejects_alignment_array_type() raises:
    with assert_raises(
        contains="general.alignment has type 9, expected UINT32"
    ):
        _ = GGUFFile(FIX + "alignment-array.gguf")


def test_accepts_explicit_alignment() raises:
    var g = GGUFFile(FIX + "alignment-valid.gguf")
    assert_equal(g.alignment, 64)
    assert_equal(g.tensor("f32").data_offset % 64, 0)
    assert_equal(g.tensor("f32").n_elements(), 8)


def test_rejects_wrapped_scalar_count_on_open() raises:
    with assert_raises(
        contains="test.array has invalid array count 4611686018427387906"
    ):
        _ = GGUFFile(FIX + "array-count-wrapped.gguf")


def test_rejects_negative_string_count_on_open() raises:
    with assert_raises(
        contains="test.strings has invalid array count 18446744073709551615"
    ):
        _ = GGUFFile(FIX + "string-count-negative.gguf")


def test_rejects_oversized_string_count_on_open() raises:
    with assert_raises(
        contains="test.strings has invalid array count 4611686018427387904"
    ):
        _ = GGUFFile(FIX + "string-count-large.gguf")


def test_accepts_string_array() raises:
    var g = GGUFFile(FIX + "string-array.gguf")
    assert_equal(len(g.get_str_array("test.strings")), 2)


def test_rejects_empty() raises:
    with assert_raises(contains="file is empty or unreadable"):
        _ = GGUFFile(FIX + "empty.gguf")


def test_rejects_short_magic() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "magic-cut.gguf")


def test_rejects_header_tensor_count() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "header-12.gguf")


def test_rejects_header_kv_count() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "header-20.gguf")


def test_rejects_metadata_cut() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "metadata-cut.gguf")


def test_rejects_oversized_key() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "key-length.gguf")


def test_rejects_key_length_that_wraps_negative() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "key-length-negative.gguf")


def test_rejects_key_length_that_wraps_the_position_sum() raises:
    # 24 + 0x7FFFFFFFFFFFFFF0 wraps to a negative Int, so a check written as
    # `at + n > size` lets it through.
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "key-length-int64max.gguf")


def test_reader_skip_and_string_refuse_a_length_that_wraps() raises:
    # 8 + (Int.MAX - 4) overflows to a negative Int.
    var buf = Bytes(16)
    var r = Reader(buf.addr, 16, 8)
    with assert_raises(contains="GGUF file truncated: wanted"):
        r.skip(Int.MAX - 4)
    assert_equal(r.at, 8)
    with assert_raises(contains="GGUF file truncated: wanted"):
        r.skip(-1)
    r.skip(8)
    assert_equal(r.at, 16)


def test_reader_refuses_a_start_outside_the_file() raises:
    var buf = Bytes(16)
    with assert_raises(contains="GGUF read position -1 is outside the file"):
        _ = Reader(buf.addr, 16, -1)
    with assert_raises(contains="GGUF read position 17 is outside the file"):
        _ = Reader(buf.addr, 16, 17)
    var at_end = Reader(buf.addr, 16, 16)
    with assert_raises(contains="GGUF file truncated"):
        _ = at_end.u8()


def test_accepts_last_read_ending_exactly_at_eof() raises:
    var g = GGUFFile(FIX + "exact-eof.gguf")
    assert_equal(g.n_tensors(), 0)


def test_rejects_version_one() raises:
    with assert_raises(contains="unsupported GGUF version 1"):
        _ = GGUFFile(FIX + "version-1.gguf")


def test_rejects_version_four() raises:
    with assert_raises(contains="unsupported GGUF version 4"):
        _ = GGUFFile(FIX + "version-4.gguf")


def test_rejects_large_tensor_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "tensors-large.gguf")


def test_rejects_negative_tensor_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "tensors-negative.gguf")


def test_rejects_large_kv_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "kv-large.gguf")


def test_rejects_negative_kv_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "kv-negative.gguf")


def test_rejects_unknown_scalar_type() raises:
    with assert_raises(contains="GGUF value type 99 has no fixed size"):
        _ = GGUFFile(FIX + "scalar-type.gguf")


def test_rejects_unknown_array_type() raises:
    with assert_raises(contains="GGUF value type 99 has no fixed size"):
        _ = GGUFFile(FIX + "array-type.gguf")


def test_rejects_nested_array() raises:
    with assert_raises(
        contains="nested GGUF arrays are not supported: test.array"
    ):
        _ = GGUFFile(FIX + "nested-array.gguf")


def test_rejects_zero_dims() raises:
    with assert_raises(contains="tensor f32 has 0 dims"):
        _ = GGUFFile(FIX + "dims-0.gguf")


def test_rejects_five_dims() raises:
    with assert_raises(contains="tensor f32 has 5 dims"):
        _ = GGUFFile(FIX + "dims-5.gguf")


def test_rejects_tensor_data_cut() raises:
    with assert_raises(contains="tensor q4k extends past the end of the file"):
        _ = GGUFFile(FIX + "tensor-cut.gguf")


def test_rejects_unsupported_row_type() raises:
    with assert_raises(contains="unsupported GGUF tensor type: IQ2_XXS"):
        _ = ggml_row_bytes(16, 256)


def test_rejects_unsupported_dequant_type() raises:
    var src = Bytes(144)
    var dst = Floats(256)
    with assert_raises(contains="unsupported GGUF tensor type: Q2_K"):
        dequant_row(10, src.addr, dst.ptr(), 256)


def test_rejects_unsupported_packed_type() raises:
    var src = Bytes(8 * 144)
    with assert_raises(contains="cannot repack weight type Q2_K"):
        _ = PackedMatrix(src.addr, 10, 8, 256)


def test_rejects_incomplete_dequant_block() raises:
    # Keep backing memory valid even when the length guard is mutated away.
    var src = Bytes(144)
    var dst = Floats(300)
    with assert_raises(contains="row length 300 is not a multiple of 256"):
        dequant_row(GGML_Q4_K, src.addr, dst.ptr(), 300)


def test_rejects_incomplete_packed_block() raises:
    var src = Bytes(8 * 144)
    with assert_raises(contains="row length 300 is not a multiple of 256"):
        _ = PackedMatrix(src.addr, GGML_Q4_K, 8, 300)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
