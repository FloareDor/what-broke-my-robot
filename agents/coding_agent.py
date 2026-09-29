"""The coding agent, running on the Gemini API through the same four tool functions the random baseline uses."""

from __future__ import annotations

import json
import os
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from agents.prompts import SYSTEM_PROMPT
from src.env import ACTUATOR_NAMES, FOOT_GEOM_NAMES
from src.experiment_api import ExperimentSession
from src.prediction import ParameterEstimate

load_dotenv()  # picks up GEMINI_API_KEY / GEMINI_MODEL from a .env file if present

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

# round trips with the model, a backstop so a confused agent can't loop forever burning API calls
MAX_TURNS = 40

# a few retries with backoff before giving up the episode over a transient server error
MAX_API_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5


def summarize_trajectory(trajectory: dict[str, np.ndarray]) -> dict:
    """Turns a raw trajectory into small JSON the model can read, instead of dumping 150+ raw timesteps."""
    if not trajectory:
        return {}

    time = trajectory["time"]
    qpos = trajectory["qpos"]
    joint_pos = qpos[:, 7:]  # first 7 entries are the free joint (xyz + quat)
    ctrl = trajectory["ctrl"]
    actuator_force = trajectory["actuator_force"]
    torso_pos = trajectory["torso_pos"]
    foot_contacts = trajectory["foot_contacts"]
    tracking_error = joint_pos - ctrl

    per_actuator = {}
    for i, name in enumerate(ACTUATOR_NAMES):
        per_actuator[name] = {
            "mean_tracking_error": float(np.mean(tracking_error[:, i])),
            "max_abs_tracking_error": float(np.max(np.abs(tracking_error[:, i]))),
            "mean_actuator_force": float(np.mean(actuator_force[:, i])),
            "max_abs_actuator_force": float(np.max(np.abs(actuator_force[:, i]))),
        }

    return {
        "n_steps": int(len(time)),
        "duration": float(time[-1] - time[0]) if len(time) > 1 else 0.0,
        "torso_start_pos": torso_pos[0].tolist(),
        "torso_end_pos": torso_pos[-1].tolist(),
        "torso_displacement_xyz": (torso_pos[-1] - torso_pos[0]).tolist(),
        "torso_min_height": float(np.min(torso_pos[:, 2])),
        "torso_max_height": float(np.max(torso_pos[:, 2])),
        "per_actuator": per_actuator,
        "foot_contact_fraction": {
            name: float(np.mean(foot_contacts[:, i])) for i, name in enumerate(FOOT_GEOM_NAMES)
        },
    }


def _tool_declarations() -> list[types.FunctionDeclaration]:
    return [
        types.FunctionDeclaration(
            name="inspect_model",
            description=(
                "Static facts about the nominal robot: actuator names/order, control "
                "ranges, home pose, nominal torso mass, nominal foot friction. Free."
            ),
            parameters_json_schema={"type": "object", "properties": {}},
        ),
        types.FunctionDeclaration(
            name="run_experiment",
            description=(
                "Run a controller on the real, unknown robot for `duration` seconds. "
                "Costs one experiment. `controller_code` is a Python source string "
                "defining `def policy(t, obs):` returning a 12-element control array "
                "(actuator order from inspect_model). `obs` has 'qpos' and 'qvel' "
                "numpy arrays. `np` is already imported, do not import it yourself."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "controller_code": {"type": "string"},
                    "duration": {"type": "number", "description": "seconds, default 3.0"},
                    "seed": {"type": "integer", "description": "default 0"},
                },
                "required": ["controller_code"],
            },
        ),
        types.FunctionDeclaration(
            name="inspect_trajectory",
            description=(
                "Summary of a past experiment: torso position/height over time, "
                "per-actuator tracking error and force, foot contact fractions. "
                "Free, call it as many times as you want on data you already have."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {"experiment_id": {"type": "integer"}},
                "required": ["experiment_id"],
            },
        ),
        types.FunctionDeclaration(
            name="submit_model",
            description=(
                "Final answer, ends the episode. Only set a field for a value you "
                "believe actually changed from nominal; leave a field out if you "
                "think it is still nominal. `friction` is the absolute foot "
                "friction coefficient (nominal 0.8). `torso_mass_scale` is a "
                "multiplier on nominal torso mass (nominal 1.0). `actuator_scale` "
                f"maps actuator name (from {ACTUATOR_NAMES}) to a strength "
                "multiplier (nominal 1.0), only for actuators you believe changed."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "friction": {"type": "number"},
                    "torso_mass_scale": {"type": "number"},
                    "actuator_scale": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                },
            },
        ),
    ]


class CodingAgent:
    """Drives one diagnosis episode against an ExperimentSession using Gemini."""

    def __init__(
        self,
        session: ExperimentSession,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        verbose: bool = True,
    ):
        self.session = session
        self.model = model
        self.verbose = verbose
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("set the GEMINI_API_KEY environment variable (or pass api_key=) to run the coding agent")
        self.client = genai.Client(api_key=api_key)
        self.transcript: list[dict] = []
        self._submitted = False

    def _log(self, kind: str, **fields) -> None:
        entry = {"kind": kind, **fields}
        self.transcript.append(entry)
        if self.verbose:
            print(f"[{kind}] {json.dumps(fields, default=str)[:600]}")

    def _dispatch(self, name: str, args: dict) -> dict:
        try:
            if name == "inspect_model":
                return self.session.inspect_model()

            if name == "run_experiment":
                if "controller_code" not in args:
                    return {"error": "missing required argument controller_code"}
                return self.session.run_experiment(
                    controller_code=args["controller_code"],
                    duration=float(args.get("duration", 3.0)),
                    seed=int(args.get("seed", 0)),
                )

            if name == "inspect_trajectory":
                trajectory = self.session.inspect_trajectory(int(args["experiment_id"]))
                return summarize_trajectory(trajectory)

            if name == "submit_model":
                estimate = ParameterEstimate(
                    friction=args.get("friction"),
                    torso_mass_scale=args.get("torso_mass_scale"),
                    actuator_scale=dict(args.get("actuator_scale") or {}),
                )
                self.session.submit_model(estimate)
                self._submitted = True
                return {"status": "submitted", "estimate": estimate.as_dict()}

            return {"error": f"unknown tool {name}"}
        except Exception as exc:  # noqa: BLE001, the agent needs to see failures and adapt, not crash the episode
            return {"error": str(exc)}

    def _generate_with_retry(self, contents: list, config: types.GenerateContentConfig):
        for attempt in range(MAX_API_RETRIES):
            try:
                return self.client.models.generate_content(model=self.model, contents=contents, config=config)
            except errors.APIError as exc:
                # 503 is overloaded, short retry. 429 is the rate limit, needs a much longer wait.
                if exc.code not in (429, 503) or attempt == MAX_API_RETRIES - 1:
                    raise
                wait = 65 if exc.code == 429 else RETRY_BACKOFF_SECONDS * (attempt + 1)
                self._log("api_retry", error=str(exc), attempt=attempt + 1, wait_seconds=wait)
                time.sleep(wait)

    def run(self) -> ParameterEstimate:
        """Runs the diagnosis loop until submit_model is called, or it runs out of turns."""
        tools = types.Tool(function_declarations=_tool_declarations())
        config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[tools])

        contents = [types.Content(role="user", parts=[types.Part.from_text(text="Begin diagnosing the robot.")])]

        for _ in range(MAX_TURNS):
            response = self._generate_with_retry(contents, config)
            candidate = response.candidates[0]
            contents.append(candidate.content)

            for part in candidate.content.parts or []:
                if getattr(part, "text", None):
                    self._log("agent_text", text=part.text)

            function_calls = response.function_calls or []
            if not function_calls:
                break

            response_parts = []
            for fc in function_calls:
                args = dict(fc.args or {})
                self._log("tool_call", name=fc.name, args=args)
                result = self._dispatch(fc.name, args)
                self._log("tool_result", name=fc.name, result=result)
                response_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))

            contents.append(types.Content(role="user", parts=response_parts))

            if self._submitted:
                break
        else:
            self._log("warning", text="hit MAX_TURNS without a submit_model call")

        if not self._submitted:
            self._log("warning", text="agent never called submit_model, defaulting to nominal (no change) estimate")
            return ParameterEstimate()

        return self.session.parameter_estimate


if __name__ == "__main__":
    # Quick manual smoke test: one agent vs one hidden robot, no baselines, no eval.
    session = ExperimentSession(seed=0, budget=8, hidden_kind="actuator_strength")
    print("Hidden perturbation (ground truth):", session.true_perturbation())
    agent = CodingAgent(session)
    estimate = agent.run()
    print("\nAgent's final estimate:", estimate.as_dict())
    print(f"Experiments used: {session.experiments_used}/{session.budget}")
