"""The gemma4 tokenizer rules on a vocabulary small enough to work out by hand.

`scripts/gen_fixtures.py` writes `tests/fixtures/gemma4-tiny/`: 279 tokens (the
five control tokens, `<|turn>`, `<turn|>`, `<|channel>` and `<|tool_response>`,
the 256 byte tokens `<0xNN>` at id 9 + NN, and a handful of text tokens) and
eight merges chosen so that each rule changes the ids when it is broken. The
real vocabulary's goldens are in `tests/test_tokenizer_gemma4.mojo`.

Ids: 265 `▁`, 266 a, 267 b, 268 c, 269 ab, 270 bc, 271 abc, 272 `▁a`, 273 `\n`,
274 `\n\n`, 275 `\n\n\n`, 276 `a\n`, 277 `▁▁`, 278 é.
Merges by rank: 0 `a b`, 1 `b c`, 2 `ab c`, 3 `▁ a`, 4 `\n \n`, 5 `a \n`,
6 `▁ ▁`, 7 `a b` again.
"""

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.tokenizer.bpe import Tokenizer

comptime DIR = "tests/fixtures/gemma4-tiny/"
comptime BYTE0 = 9


def _show(ids: List[Int]) -> String:
    var s = String("[")
    for i in range(len(ids)):
        s += (", " if i else "") + String(ids[i])
    return s + "]"


def _check(mut tok: Tokenizer, text: String, want: List[Int]) raises:
    var got = tok.encode(text)
    var same = len(got) == len(want)
    if same:
        for i in range(len(got)):
            if got[i] != want[i]:
                same = False
    assert_true(
        same,
        "encode(" + repr(text) + ") = " + _show(got) + ", want " + _show(want),
    )


def _tok(name: String) raises -> Tokenizer:
    return Tokenizer(GGUFFile(DIR + name))


def test_tiny_bos_is_added_although_the_file_says_false() raises:
    var tok = _tok("base.gguf")
    assert_true(tok.add_bos)
    _check_with_bos(tok, "a", [2, 266])
    _check(tok, "a", [266])


def _check_with_bos(mut tok: Tokenizer, text: String, want: List[Int]) raises:
    var got = tok.encode(text, add_bos=tok.add_bos)
    assert_equal(_show(got), _show(want))


def test_tiny_eos_is_added_only_when_the_file_asks() raises:
    var tok = _tok("eos.gguf")
    _check_with_bos(tok, "a", [2, 266, 1])
    _check(tok, "a", [266])
    var base = _tok("base.gguf")
    _check_with_bos(base, "a", [2, 266])


def test_tiny_merges_apply_lowest_rank_first() raises:
    var tok = _tok("base.gguf")
    # a b (rank 0) and b c (rank 1) compete: ab first, then ab c (rank 2).
    _check(tok, "abc", [271])
    _check(tok, "ab", [269])
    _check(tok, "bc", [270])
    _check(tok, "abab", [269, 269])


def test_tiny_equal_ranks_merge_leftmost_first() raises:
    var tok = _tok("base.gguf")
    # ▁ ▁ at positions 0-1 and 1-2 share rank 6; the left one goes first.
    _check(tok, "   ", [277, 265])


def test_tiny_the_first_of_two_equal_merges_keeps_its_rank() raises:
    var tok = _tok("base.gguf")
    assert_equal(tok.pair_rank("a", "b"), 0)
    assert_equal(tok.pair_rank("b", "c"), 1)
    assert_equal(tok.pair_rank("a", "c"), -1)
    # With "a b" at rank 7 instead, b c would go first and abc would not form.
    _check(tok, "abc", [271])


def test_tiny_spaces_become_lower_blocks() raises:
    var tok = _tok("base.gguf")
    _check(tok, "a b", [266, 265, 267])
    # No space is added in front: "a" is "a", not "▁a".
    _check(tok, "a", [266])
    _check(tok, " a", [272])
    # ▁ ▁ (rank 6) loses to ▁ a (rank 3): the space run splits before the a.
    _check(tok, "  a", [265, 272])


def test_tiny_a_newline_run_that_is_a_token_stays_whole() raises:
    var tok = _tok("base.gguf")
    # \n\n\n is a token. Merging would give \n\n then \n (leftmost first).
    _check(tok, "\n\n\n", [275])
    _check(tok, "\n\n", [274])
    _check(tok, "\n", [273])


def test_tiny_a_newline_run_that_is_not_a_token_is_merged() raises:
    var tok = _tok("base.gguf")
    # Four newlines are no token: \n\n and \n\n, leftmost merge first.
    _check(tok, "\n\n\n\n", [274, 274])


def test_tiny_no_merge_crosses_a_newline() raises:
    var tok = _tok("base.gguf")
    # "a \n" (rank 5) exists, but the pre-split puts a and \n in separate words.
    _check(tok, "a\n", [266, 273])
    _check(tok, "a\nb", [266, 273, 267])
    # A carriage return is not a newline for the split: it travels with its
    # word, and here it has no token, so it is a byte.
    _check(tok, "\r\n", [BYTE0 + 13, 273])


def test_tiny_characters_without_a_token_become_byte_tokens() raises:
    var tok = _tok("base.gguf")
    _check(tok, "x", [BYTE0 + 0x78])
    # ü is C3 BC, and has no token; é has one.
    _check(tok, "ü", [BYTE0 + 0xC3, BYTE0 + 0xBC])
    _check(tok, "é", [278])
    _check(tok, "éx", [278, BYTE0 + 0x78])
    assert_equal(tok.decode([BYTE0 + 0xC3, BYTE0 + 0xBC]), "ü")


def test_tiny_special_tokens_in_the_text_become_their_ids() raises:
    var tok = _tok("base.gguf")
    _check(tok, "<|turn>a<turn|>", [5, 266, 6])
    _check(tok, "<bos>", [2])
    _check(tok, "<|channel>", [7])
    _check(tok, "a<mask>b", [266, 4, 267])
    # Not a marker: it is cut into characters and merged like any text.
    assert_true(len(tok.encode("<|turn")) > 1)


def test_tiny_decode_follows_the_token_attributes() raises:
    var tok = _tok("base.gguf")
    assert_equal(tok.decode([5, 266, 6]), "a")
    assert_equal(tok.decode([5, 266, 6], skip_special=False), "<|turn>a<turn|>")
    # User-defined <|channel> prints; <|tool_response> was made control.
    assert_equal(tok.decode([7, 266]), "<|channel>a")
    assert_equal(tok.decode([8, 266]), "a")
    assert_equal(tok.decode([8, 266], skip_special=False), "<|tool_response>a")
    assert_equal(tok.decode([265, 266, 277]), " a  ")
    assert_equal(tok.eot_id, 6)
    assert_true(tok.is_eog(1) and tok.is_eog(6) and tok.is_eog(8))
    assert_true(not tok.is_eog(7))


def test_tiny_a_missing_byte_token_is_named() raises:
    with assert_raises(contains="no byte token <0x78>"):
        _ = _tok("no-byte-token.gguf")


def test_tiny_a_byte_token_with_the_wrong_text_is_named() raises:
    with assert_raises(contains="token 129 has type byte"):
        _ = _tok("bad-byte-text.gguf")


def test_tiny_a_merge_without_a_space_is_named() raises:
    with assert_raises(contains="tokenizer.ggml.merges[3]"):
        _ = _tok("merge-without-space.gguf")


def test_tiny_token_and_type_counts_must_agree() raises:
    with assert_raises(
        contains="tokenizer.ggml.token_type has 278 entries but"
    ):
        _ = _tok("short-types.gguf")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
