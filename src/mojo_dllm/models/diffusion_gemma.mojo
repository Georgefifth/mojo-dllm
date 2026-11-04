"""DiffusionGemma on the CPU: layer kinds, region masks and the forward pass.

Every rule here was read from llama.cpp's `src/models/diffusion-gemma.cpp` and
`src/models/gemma4-common.h` (ggml-org/llama.cpp#24423 at 2e715c0639), not from
the issue text; `scripts/dg_numpy.py` repeats them in numpy for the tests.
"""


@fieldwise_init
struct KeyRange(Copyable, ImplicitlyCopyable, Movable):
    """Keys [lo, hi) a query may attend to. Every allowed set is an interval."""

    var lo: Int
    var hi: Int


def key_range(
    q: Int, n_prompt: Int, window: Int, sliding: Bool, n_keys: Int
) -> KeyRange:
    """The keys query `q` (an absolute position) attends to in one layer.

    The first `n_prompt` positions are the prompt, the rest the canvas.
    A prompt row is causal over the prompt, clipped to `window` keys in a
    sliding layer (it sees q - window + 1 .. q). A canvas row sees the whole
    canvas, and of the prompt everything in a global layer but only the last
    window - 1 tokens in a sliding one (the branch's `canvas_prompt_lo`).
    """
    if q < n_prompt:
        if sliding:
            return KeyRange(max(0, q - window + 1), q + 1)
        return KeyRange(0, q + 1)
    if sliding:
        return KeyRange(max(0, n_prompt - window + 1), n_keys)
    return KeyRange(0, n_keys)
