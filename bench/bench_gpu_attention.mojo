"""Dream-shaped GQA: original and wide cooperative tiles, seeded inputs.

Build with scripts/build.sh bench/bench_gpu_attention.mojo build/bench-attention.
Run build/bench-attention CANVAS_LENGTH on an idle GPU. Round zero is warmup.
Timings include host enqueue and synchronization. Output must match bitwise.
"""
from std.sys import argv
from std.time import perf_counter_ns
from std.math import sqrt
from max.gpu.host import DeviceContext
from mojo_dllm.sys.mem import Floats, F32Ptr
from mojo_dllm.gpu.kernels import dev_f32, attention_gpu
from mojo_dllm.gpu.kernels import _attention_kernel


def measure[
    BK: Int, BQ: Int
](
    ctx: DeviceContext,
    q: F32Ptr,
    k: F32Ptr,
    v: F32Ptr,
    o: F32Ptr,
    L: Int,
    reps: Int,
) raises -> Float64:
    ctx.synchronize()
    var start = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_function[_attention_kernel[128, BK, BQ]](
            q,
            k,
            v,
            o,
            Int32(L),
            Int32(L),
            Int32(28),
            Int32(4),
            Float32(1.0 / sqrt(128.0)),
            grid_dim=((L + BQ - 1) // BQ, 28),
            block_dim=BQ * 2,
        )
    ctx.synchronize()
    return Float64(perf_counter_ns() - start) / 1e6 / Float64(reps)


def main() raises:
    var args = argv()
    var L = Int(args[1]) if len(args) > 1 else 1051
    var ctx = DeviceContext()
    var hq = Floats(L * 28 * 128)
    var hk = Floats(L * 4 * 128)
    var hv = Floats(L * 4 * 128)
    var seed = UInt64(42)
    for i in range(hq.n):
        seed = seed * 6364136223846793005 + 1
        hq[i] = Float32(Int((seed >> 33) % 2001) - 1000) / 1000.0
    for i in range(hk.n):
        seed = seed * 6364136223846793005 + 1
        hk[i] = Float32(Int((seed >> 33) % 2001) - 1000) / 1000.0
        seed = seed * 6364136223846793005 + 1
        hv[i] = Float32(Int((seed >> 33) % 2001) - 1000) / 1000.0
    var q = ctx.enqueue_create_buffer[DType.float32](hq.n)
    var k = ctx.enqueue_create_buffer[DType.float32](hk.n)
    var v = ctx.enqueue_create_buffer[DType.float32](hv.n)
    var o = ctx.enqueue_create_buffer[DType.float32](hq.n)
    ctx.enqueue_copy(q, hq.ptr())
    ctx.enqueue_copy(k, hk.ptr())
    ctx.enqueue_copy(v, hv.ptr())
    _ = measure[32, 64](
        ctx, dev_f32(q), dev_f32(k), dev_f32(v), dev_f32(o), L, 1
    )
    var reference = Floats(hq.n)
    var actual = Floats(hq.n)
    ctx.enqueue_copy(reference.ptr(), o)
    ctx.synchronize()
    for round in range(5):
        for order in range(2):
            var candidate = (order + round) % 2 == 1
            var ms = Float64(0)
            if candidate:
                _ = measure[128, 128](
                    ctx, dev_f32(q), dev_f32(k), dev_f32(v), dev_f32(o), L, 5
                )
                ms = measure[128, 128](
                    ctx, dev_f32(q), dev_f32(k), dev_f32(v), dev_f32(o), L, 30
                )
            else:
                _ = measure[32, 64](
                    ctx, dev_f32(q), dev_f32(k), dev_f32(v), dev_f32(o), L, 5
                )
                ms = measure[32, 64](
                    ctx, dev_f32(q), dev_f32(k), dev_f32(v), dev_f32(o), L, 30
                )
            ctx.enqueue_copy(actual.ptr(), o)
            ctx.synchronize()
            var error = Float32(0)
            for i in range(hq.n):
                error = max(error, abs(actual[i] - reference[i]))
            print(
                '{"L":',
                L,
                ',"candidate":',
                Int(candidate),
                ',"round":',
                round,
                ',"ms":',
                ms,
                ',"max_abs":',
                error,
                "}",
            )
