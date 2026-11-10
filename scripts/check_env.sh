#!/usr/bin/env bash
# The MOJO_DLLM_* wiring in main() is only observable through the real entry
# path: the Mojo tests call execute() directly, so they never see getenv or
# with_env. This drives the built binary the way a user would and pins each
# piece of the wiring (#122): the env merge, the (from VAR) suffix, the
# stderr-only --verbose report, and command-line precedence.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
bin=${1:-build/mojo-dllm}
model=tests/fixtures/tiny-llada.gguf
if [ ! -x "$bin" ]; then
    echo "FAIL: $bin is not built" >&2
    exit 1
fi

# run_env VAR=value ARGS... -> sets globals rc/out/err (stdout, stderr split).
run_env() {
    local var=$1
    shift
    out="$(env "$var" "$bin" run --model "$model" --prompt hi "$@" 2>/tmp/env_err.$$)"
    rc=$?
    err="$(cat /tmp/env_err.$$)"
    rm -f /tmp/env_err.$$
}

fail() {
    printf 'FAIL: %s\n  rc=%s\n  stdout: %s\n  stderr: %s\n' \
        "$1" "$rc" "$out" "$err" >&2
    exit 1
}

# 1. A bad value from a variable is a usage error that names the variable.
#    Kills: main() dropping the with_env merge; execute dropping env_problem;
#    main() not passing from_env.
run_env MOJO_DLLM_STEPS=abc
[ "$rc" -eq 2 ] || fail "env-supplied bad --steps: expected rc 2"
grep -Fq "flag --steps needs an integer, got 'abc' (from MOJO_DLLM_STEPS)" <<<"$err" \
    || fail "env-supplied bad --steps: stderr lacks the (from VAR) suffix"
[ -z "$out" ] || fail "env-supplied bad --steps: stdout must stay empty"
echo 'ok: a bad env value is a usage error naming its variable'

# 2. A flag on the command line wins over the variable (here the variable is
#    bad, the flag is good: the run must reach the model).
run_env MOJO_DLLM_STEPS=abc --steps 4
[ "$rc" -eq 1 ] || fail "--steps 4 override: expected rc 1 (model error)"
grep -Fq 'required metadata key missing' <<<"$err" \
    || fail "--steps 4 override: did not reach the model"
echo 'ok: the command line beats the variable'

# 3. A typed flag left without a value is blamed on the user, not on a
#    variable that happens to be set (the issue's trap).
run_env MOJO_DLLM_STEPS=8 --steps
[ "$rc" -eq 2 ] || fail "trailing --steps: expected rc 2"
grep -Fq 'flag --steps needs a value' <<<"$err" \
    || fail "trailing --steps: wrong message"
grep -Fq '(from' <<<"$err" && fail "trailing --steps: must not blame the variable"
echo 'ok: needs-a-value stays unsuffixed'

# 4. run --verbose reports each env-supplied flag on stderr only.
#    Kills: removing the report; sending it to stdout.
run_env MOJO_DLLM_THREADS=2 --verbose
grep -Fxq 'from environment: --threads 2 (MOJO_DLLM_THREADS)' <<<"$err" \
    || fail "--verbose: stderr lacks the from-environment line"
grep -Fq 'from environment' <<<"$out" \
    && fail "--verbose: the report must not appear on stdout"
echo 'ok: --verbose reports env flags on stderr only'

# 5. threshold is validated too: a bad value names its variable (review #2).
run_env MOJO_DLLM_THRESHOLD=abc
[ "$rc" -eq 2 ] || fail "env-supplied bad --threshold: expected rc 2"
grep -Fq "flag --threshold needs a finite number, got 'abc' (from MOJO_DLLM_THRESHOLD)" <<<"$err" \
    || fail "env-supplied bad --threshold: stderr lacks the (from VAR) suffix"
echo 'ok: a bad --threshold value names its variable'

# 6. An empty variable counts as unset.
run_env MOJO_DLLM_STEPS=
[ "$rc" -eq 1 ] || fail "empty MOJO_DLLM_STEPS: expected rc 1 (model error)"
grep -Fq 'required metadata key missing' <<<"$err" \
    || fail "empty MOJO_DLLM_STEPS: did not reach the model"
grep -Fq '(from MOJO' <<<"$err" && fail "empty variable must not be reported"
echo 'ok: an empty variable counts as unset'
