"""A running vLLM server as a GSM8K engine (issue #179): a client for its OpenAI-compatible endpoint.

The engine never starts or stops a server and loads no weights. It asks a server somebody else started, over HTTP
(`--base-url`), to answer each problem with `POST /v1/chat/completions`: one user message holding the same prompt
text the other engines use, `temperature` 0 and `max_tokens` equal to the generation length. The server applies the
model's chat template, so the template is the server's and `flags` says so.

Field names were read from vLLM's source (main, 2026-11-09), not from memory:

- `GET /version` answers `{"version": ...}` (vllm/entrypoints/serve/instrumentator/basic.py).
- `GET /v1/models` answers one card per served name with `id`, `root` (the model path given to `vllm serve`) and
  `max_model_len` (vllm/entrypoints/openai/models/serving.py).
- A diffusion model exposes the Prometheus counter `vllm:diffusion_num_denoising_steps_total` on `/metrics`
  (vllm/v1/spec_decode/metrics.py: "Number of denoising steps", the spec-decode counters under diffusion names).
  It is server-wide, not per request. With one client and one request at a time (batch size 1), its change across
  one request is that request's step count. The adapter reads it before and after each generation. When the
  server has no such counter, `forward_passes` is null; the engine never invents a count.

What the server cannot tell the client comes from options and is recorded as declared, not verified:
`--quantization` (required, and part of the arm's `model` so `compare --across-weights` sees different weights),
`--attention-backend` and `--chat-template`. `--checkpoint` is compared with the `root` the server reports and a
difference is refused.

The isolation checks (`bench/isolation.py`) run on the CLIENT machine only. A remote server's GPU, clocks and
neighbours are not observed, and `flags` says `"server": "remote"` so nobody reads them as the server's.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

from . import EngineError, GenConfig, register
from ._http import HttpClient, HttpError

STEPS_METRIC = "vllm:diffusion_num_denoising_steps_total"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024    # a 256-token completion is a few KB; /metrics is the largest answer, well under this
CLIENT_RAM_GB = 0.5         # my estimate for the Python loop of a client that holds no model; not a measurement
_SAMPLE = re.compile(r"^(?P<name>[^\s{]+)(?:\{[^}]*\})?\s+(?P<value>\S+)", re.M)


def parse_steps_total(text: str) -> int | None:
    """The counter summed over its label sets (one per engine), or None when the page has no such counter."""
    total, seen = 0.0, False
    for m in _SAMPLE.finditer(text):
        if m.group("name") == STEPS_METRIC:
            try:
                total += float(m.group("value"))
            except ValueError:
                raise EngineError(f"/metrics has an unreadable value for {STEPS_METRIC}: {m.group('value')!r}") from None
            seen = True
    return int(total) if seen else None


@register
class VLLM:
    name = "vllm"

    def __init__(self, args: argparse.Namespace) -> None:
        self.base_url = args.base_url
        self.served_model = args.served_model
        self.checkpoint = args.checkpoint
        self.quantization = args.quantization
        self.attention_backend = args.attention_backend
        self.chat_template = args.chat_template
        self.use_env_proxy = args.use_env_proxy
        self.client: HttpClient | None = None
        self._vllm_version: str | None = None
        self._root: str | None = None
        self._steps_measured = False

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                            help="the running vLLM server, without /v1 (default %(default)s). Never put credentials in it")
        parser.add_argument("--served-model", default=None,
                            help="the model name the server serves (`--served-model-name`, else the model path); "
                                 "required, and the run is refused when /v1/models does not list it")
        parser.add_argument("--quantization", default=None,
                            help="the checkpoint's quantization as you started the server (bf16, fp8-dynamic, nvfp4); "
                                 "required. The server does not report it, so it is recorded as declared")
        parser.add_argument("--checkpoint", default=None,
                            help="the checkpoint you passed to `vllm serve`; refused when it differs from the `root` "
                                 "the server reports")
        parser.add_argument("--attention-backend", default=None,
                            help="the attention backend you started the server with (TRITON_ATTN, FLASH_ATTN); "
                                 "recorded as declared")
        parser.add_argument("--chat-template", default=None,
                            help="the --chat-template you started the server with, if any; recorded as declared "
                                 "(the server applies it, the client cannot read it)")
        parser.add_argument("--use-env-proxy", action="store_true",
                            help="honour http_proxy and friends; off by default so no proxy sits in the timed path")

    def model_name(self, cfg: GenConfig) -> str:
        """The `model` of every row: what `compare` reads to decide whether the weights differ, so the
        quantization is part of it."""
        return f"{pathlib.Path(cfg.model).name}|{self.quantization}"

    def ram_gb_after_load(self, cfg: GenConfig) -> float:
        """The preflight sizes memory from a model file; this client has none. It asks for a small fixed amount."""
        return CLIENT_RAM_GB

    def check(self, cfg: GenConfig) -> None:
        if cfg.device not in ("cpu", "gpu"):
            raise SystemExit(f"vllm runs with --device cpu or gpu (the isolation checks of THIS machine), not {cfg.device!r}")
        if cfg.extra:
            raise SystemExit(f"vllm takes no extra engine flags; got {' '.join(cfg.extra)}")
        if cfg.steps != cfg.gen_length:
            raise SystemExit("vllm takes its step rule from the server (`--diffusion-config` at serve time); "
                             f"--steps {cfg.steps} would be ignored, so it is refused")
        if not self.served_model:
            raise SystemExit("--served-model is required: the model name the server serves")
        if not self.quantization:
            raise SystemExit("--quantization is required (bf16, fp8-dynamic, nvfp4): the server does not report it, "
                             "and it is part of the arm's model so compare sees different weights")
        self.client = HttpClient(self.base_url, timeout_s=cfg.timeout_s, max_bytes=MAX_RESPONSE_BYTES,
                                use_env_proxy=self.use_env_proxy)
        cards = self._model_cards()
        ids = [c["id"] for c in cards]
        if self.served_model not in ids:
            raise EngineError(f"the server reports the models {', '.join(ids)}; --served-model is {self.served_model!r}")
        root = next(c for c in cards if c["id"] == self.served_model).get("root")
        self._root = root if isinstance(root, str) else None
        if self.checkpoint is not None and self._root is not None and self.checkpoint != self._root:
            raise EngineError(f"--checkpoint is {self.checkpoint!r} but the server reports root {self._root!r} for "
                              f"{self.served_model!r}: the arm would be labelled with weights it did not run")
        doc = self.client.get_json("/version")
        ver = doc.get("version") if isinstance(doc, dict) else None
        if not isinstance(ver, str) or not ver:
            raise EngineError(f"GET /version did not answer a version string: {doc!r}")
        self._vllm_version = ver
        self._steps_measured = self._read_steps() is not None

    def _model_cards(self) -> list[dict]:
        doc = self.client.get_json("/v1/models")                     # type: ignore[union-attr]
        data = doc.get("data") if isinstance(doc, dict) else None
        if not isinstance(data, list):
            raise EngineError(f"GET /v1/models answered a malformed model list: {doc!r}")
        if not data:
            raise EngineError("the server reports no models")
        if not all(isinstance(c, dict) and isinstance(c.get("id"), str) for c in data):
            raise EngineError(f"GET /v1/models answered a malformed model list: {doc!r}")
        return data

    def _read_steps(self) -> int | None:
        """The denoising-step counter, or None when the server has no /metrics or no such counter. Any other
        failure raises: an unreadable page is not an absent counter."""
        try:
            text = self.client.get_text("/metrics")                  # type: ignore[union-attr]
        except HttpError as e:
            if e.status == 404:
                return None
            raise
        return parse_steps_total(text)

    def version(self) -> str:
        """JSON: the release the server reports (the only field it can tell), then the declared fields."""
        return json.dumps({"vllm": self._vllm_version, "attention_backend": self.attention_backend,
                           "checkpoint": self.checkpoint or self._root, "quantization": self.quantization},
                          sort_keys=True)

    def flags(self, cfg: GenConfig) -> dict:
        declared = [k for k, v in (("attention_backend", self.attention_backend),
                                   ("checkpoint", self.checkpoint if self._root is None else None),
                                   ("quantization", self.quantization)) if v]
        return {"engine": self.name, "server": "remote", "device": cfg.device, "gen_length": cfg.gen_length,
                "temperature": 0.0, "served_model": self.served_model, "quantization": self.quantization,
                "declared_not_verified": declared,
                "checkpoint_reported_by_server": self._root,
                "checkpoint_matches_server": (self.checkpoint == self._root) if self.checkpoint and self._root else None,
                "chat_template": {"applied_by": "server", "declared": self.chat_template, "verified": False},
                "forward_passes_measured": self._steps_measured,
                "forward_passes_source": STEPS_METRIC + " (change across one request)" if self._steps_measured else None}

    def generate(self, prompt: str, cfg: GenConfig) -> dict:
        before = self._read_steps() if self._steps_measured else None
        body = {"model": self.served_model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": cfg.gen_length, "temperature": 0, "stream": False}
        start = time.perf_counter()
        out = self.client.post_json("/v1/chat/completions", body)    # type: ignore[union-attr]
        generate_s = time.perf_counter() - start
        text, prompt_tokens, completion_tokens, finish = self._parse(out)
        steps = None
        if self._steps_measured:
            after = self._read_steps()
            if before is None or after is None or after - before <= 0:
                raise EngineError(f"{STEPS_METRIC} went from {before} to {after} across one request: the server "
                                  "reports no denoising steps for it (another client, or the counter is not "
                                  "this model's), so the step count would be wrong")
            steps = after - before
        return dict(text=text, generate_s=generate_s, forward_passes=steps, steps=steps,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, finish_reason=finish)

    def _parse(self, out: object) -> tuple[str, int, int, object]:
        if not isinstance(out, dict):
            raise EngineError(f"the completion is not a JSON object: {out!r}")
        if out.get("model") != self.served_model:
            raise EngineError(f"the response names the model {out.get('model')!r}, not {self.served_model!r}")
        choices = out.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise EngineError(f"the completion has no choices: {out!r}")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise EngineError(f"the server returned an empty completion (finish_reason {choices[0].get('finish_reason')!r})")
        usage = out.get("usage")
        counts = [usage.get(k) if isinstance(usage, dict) else None for k in ("prompt_tokens", "completion_tokens")]
        if not all(isinstance(n, int) and not isinstance(n, bool) for n in counts) or counts[1] <= 0:
            raise EngineError(f"the completion carries no usable token counts: {usage!r}")
        return content, counts[0], counts[1], choices[0].get("finish_reason")
