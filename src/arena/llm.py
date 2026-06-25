"""
LLM planner for the arena -- a plug-and-play planner driven by any chat LLM.

Design goals:
  * Provider-agnostic. The planner talks to a tiny `LLMClient` interface
    (`complete(system, user) -> text`). Swapping Claude for another provider is a
    new adapter, not a new planner. `AnthropicClient` is the bundled adapter.
  * Drop-in. `LLMPlanner` implements the same `Planner` protocol as the scripted
    baselines, so the harness, environment, and evaluator are untouched.
  * Robust. The model's reply is JSON-extracted, validated against the live
    observation (no hallucinated objects/regions), re-asked on failure, and
    finally falls back to a safe `stop` -- a bad reply never crashes a run.
  * Anti-bias. Every model gets the IDENTICAL system prompt + observation; the
    task-specific goal lives in the observation, not the prompt. Nothing here is
    tuned per task or per model.

Single API call per step (stateless): the observation already carries the
history of what was tried, so each decision is reproducible from the prompt.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .contracts import Action, ActionParseError, Observation, TaskSpec, parse_action

# Task-agnostic. Explains the role, the action vocabulary, and the output
# contract -- NOT how to solve any particular task (that would defeat the
# benchmark). The goal text is supplied per-step in the observation.
SYSTEM_PROMPT = """\
You are the high-level planner for a robot arm doing tabletop manipulation.

Each turn you are given the current scene, the goal, and the history of actions
already attempted this episode. Choose the single best NEXT action toward the
goal. A low-level controller executes it and you will see the updated scene next
turn, so you can react to what worked or failed.

You may emit ONLY these actions, as a single JSON object:
  {"action": "place", "object": "<id>", "region": "<region>"}  -- pick <object>, put it in a named region
  {"action": "stack", "object": "<id>", "on": "<base_id>"}     -- pick <object>, place it on top of <base_id>
  {"action": "stop", "reason": "<short reason>"}               -- goal achieved, or no useful action remains

Rules:
- Use ONLY object ids and region names that appear in the observation.
- Each object reports: reach (OK/BORDERLINE/HARD), obstacle (CLEAR/NEAR/TOO_CLOSE),
  held, and placed (already satisfies the goal). An object that is HARD to reach
  or TOO_CLOSE to an obstacle usually cannot be picked -- prefer others.
- Reason from the goal text; do not assume a fixed strategy.
- Respond with ONLY the JSON object. No prose, no markdown, no code fences.
"""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat interface. An adapter per provider implements this."""

    name: str

    def complete(self, system: str, user: str) -> str: ...


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


class AnthropicClient:
    """Claude adapter via the official `anthropic` SDK.

    Adaptive thinking is on by default (next-action planning is real reasoning,
    which is what the benchmark measures); disable for cheaper/faster runs. The
    SDK auto-retries 429/5xx with backoff. Reads ANTHROPIC_API_KEY from the env.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        thinking: bool = True,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - env dependent
            raise LLMError("the 'anthropic' package is required: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.name = model
        self.thinking = thinking
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        resp = self._client.messages.create(**kwargs)
        if getattr(resp, "stop_reason", None) == "refusal":
            raise LLMError("model refused the request")
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


class OpenAIClient:
    """OpenAI adapter via the official `openai` SDK (Chat Completions).

    JSON mode (`response_format={"type":"json_object"}`) is on by default since
    the system prompt asks for a JSON object -- it makes the reply parse cleanly.
    Targets standard chat models (gpt-4o, gpt-4.1, gpt-4o-mini). For reasoning
    models (o-series) you may need to turn json_mode off and/or adjust the token
    cap; everything else in the planner is unchanged. Reads OPENAI_API_KEY.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        max_tokens: int = 2048,
        json_mode: bool = True,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - env dependent
            raise LLMError("the 'openai' package is required: pip install openai") from exc
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.name = model
        self.max_tokens = max_tokens
        self.json_mode = json_mode

    def complete(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()


class ScriptedClient:
    """Test/baseline client: returns canned replies, no API. `responder` is a
    callable (system, user) -> reply text, so a test can simulate a model
    deterministically and exercise the full parse/validate/act path offline."""

    def __init__(self, responder: Callable[[str, str], str], name: str = "scripted-fake") -> None:
        self._responder = responder
        self.name = name

    def complete(self, system: str, user: str) -> str:
        return self._responder(system, user)


def infer_provider(model: str) -> str:
    """Guess the provider from a model id (claude-* -> anthropic, gpt-*/o* -> openai)."""
    m = model.lower()
    if m.startswith(("claude", "anthropic")):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt")) or "openai" in m:
        return "openai"
    return "anthropic"


def make_llm_client(
    model: str,
    provider: Optional[str] = None,
    *,
    thinking: bool = True,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
) -> LLMClient:
    """Build the right adapter for a model. `provider` overrides the guess from
    the model name -- so a runner only needs `--model` (and optional --provider)."""
    provider = (provider or infer_provider(model)).lower()
    if provider == "openai":
        return OpenAIClient(model=model, api_key=api_key)
    if provider == "anthropic":
        return AnthropicClient(model=model, thinking=thinking, max_tokens=max_tokens, api_key=api_key)
    raise LLMError(f"unknown provider '{provider}' (use 'anthropic' or 'openai')")


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #


def extract_json_object(text: str) -> dict:
    """Pull the first balanced {...} object out of a model reply (tolerates code
    fences or stray prose around it)."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in reply")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON object in reply")


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


class LLMPlanner:
    """Closed-loop planner driven by an LLM. Same interface as the scripted
    baselines; the only difference is judgement instead of hard-coded rules."""

    def __init__(
        self,
        client: LLMClient,
        *,
        name: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_retries: int = 3,
        log_event: Optional[Callable[..., None]] = None,
    ) -> None:
        self.client = client
        self.name = name or f"llm:{getattr(client, 'name', 'model')}"
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self._log = log_event or (lambda *a, **k: None)

    def reset(self, task: TaskSpec, obs: Observation) -> None:  # noqa: ARG002
        return None

    def act(self, obs: Observation) -> Action:
        base_user = self._build_user(obs)
        correction = ""
        last_error = "no_reply"
        for attempt in range(self.max_retries):
            try:
                reply = self.client.complete(self.system_prompt, base_user + correction)
            except Exception as exc:  # client/network/refusal -- give it another go
                last_error = f"client_error:{exc}"
                correction = "\n\nThe previous attempt errored. Reply with ONLY a valid JSON action."
                continue
            try:
                action = self._parse_and_validate(reply, obs)
                self._log("LLM_ACTION", "OK", planner=self.name, step=obs.step,
                          attempt=attempt, action=action.describe())
                return action
            except (ActionParseError, ValueError) as exc:
                last_error = str(exc)
                correction = (
                    f"\n\nYour previous reply was invalid: {exc}. "
                    "Reply with ONLY one JSON action object using ids/regions from the observation."
                )
        self._log("LLM_ACTION", "FAILED", planner=self.name, step=obs.step, failure_reason=last_error)
        return Action.stop(reason=f"llm_no_valid_action:{last_error}")

    # -- internals --------------------------------------------------------- #

    def _build_user(self, obs: Observation) -> str:
        payload = json.dumps(obs.to_dict(), indent=2)
        return (
            f"Goal: {obs.goal}\n\n"
            f"Observation (JSON):\n{payload}\n\n"
            "What is your single next action? Respond with ONLY the JSON object."
        )

    def _parse_and_validate(self, reply: str, obs: Observation) -> Action:
        action = parse_action(extract_json_object(reply))
        if action.type.value == "stop":
            return action
        known = obs.objects_by_id
        if action.object_id not in known:
            raise ActionParseError(f"unknown object '{action.object_id}'")
        if action.type.value == "stack" and action.on not in known:
            raise ActionParseError(f"unknown base object '{action.on}'")
        if action.type.value == "place" and action.region not in obs.regions:
            raise ActionParseError(f"unknown region '{action.region}'")
        return action
