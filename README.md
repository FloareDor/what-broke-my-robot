# what broke my robot?

![the robot](report/renders/hero_standing.png)

Can an AI agent figure out what physically changed on a robot, just by running its own experiments on it, better than doing nothing and better than testing at random?

I take a simulated Unitree Go1 quadruped in MuJoCo, secretly change one physical thing about it (friction, torso mass, or one leg motor's strength), and see if an agent can poke at it, read the data, and figure out what changed. Then I check if its guess actually predicts the robot better than doing nothing, and better than just probing around randomly.

## why I reused so much stuff

The interesting part is the agent doing diagnosis, not a simulator or an optimizer. So I grabbed what already exists:

- **robot**: Go1 from MuJoCo Menagerie, a submodule in `third_party/`.
- **controller**: the model's own standing pose, plus a hand-coded sine-wave trot on top. No training.
- **fitting**: `scipy.optimize.least_squares`.
- **agent**: Gemini API, plain function calling, no framework.

Everything else (env, perturbations, experiment API, agent loop) is mine.

## how it works

One round, an episode, works like this. I keep the fault itself simple: exactly one physical property changes per episode, and the agent gets a fixed budget of 8 experiments to figure out which one.

1. **Healthy robot.** Walking normally. All the agent is told about.
2. **Something breaks.** I secretly change grip, weight, or one leg motor's strength.
3. **It investigates.** It writes and runs its own test movements, up to 8 times.
4. **It answers.** It submits an estimate, for example that the left leg is at 80% strength.
5. **I grade it.** I test the estimate against new movements it never saw during testing.

![healthy vs faulty robot comparison](report/renders/healthy_vs_faulty.png)

Same walking command and starting position, shown at three points in time. Top row is healthy. Bottom row has one rear hip motor cut to 35% strength (exaggerated here for visibility). By 2.5 seconds it has tipped over.

### what the agent actually has access to

It never sees an image, a video, or a 3D scene, only four functions and whatever numbers they return:

- `inspect_model()`: static facts about the healthy robot (joint names, movement limits, standing pose). Free to call.
- `run_experiment(code, duration)`: the agent writes control code and runs it on the real, possibly broken robot. Costs 1 of its 8 tries.
- `inspect_trajectory(id)`: a numeric summary of a past run (body position over time, per-joint tracking error and force, which feet touched ground). Free, the agent can call it as many times as it wants.
- `submit_model(estimate)`: its final answer. Ends the episode.

Here's what `inspect_trajectory` actually returns after one real run against a robot with a weakened rear hip motor, no labels, no hint anything is wrong, just numbers:

```json
{
  "n_steps": 150,
  "torso_displacement_xyz": [0.061, 0.030, 0.007],
  "torso_min_height": 0.252,
  "torso_max_height": 0.306,
  "per_actuator": {
    "RR_hip":   { "mean_tracking_error": -0.004, "max_abs_tracking_error": 0.073 },
    "FR_thigh": { "mean_tracking_error":  0.041, "max_abs_tracking_error": 0.244 }
  },
  "foot_contact_fraction": { "FR": 0.65, "FL": 0.60, "RR": 0.60, "RL": 0.76 }
}
```

Figuring out from tables like this that a specific rear hip motor is 18% weak, in 8 tries or fewer.

## how it's laid out

```
third_party/mujoco_menagerie/   the Go1 model, pulled in as a submodule
src/
  env.py               thin wrapper around the mujoco model, step/reset/etc
  controller.py        standing pose + open loop trot gait
  perturbations.py     the hidden fault, applied to a cloned model
  prediction.py        turns an agent's guess into an actual model
  metrics.py           held out prediction error, nominal vs guess vs real
  experiment_api.py    the tool interface an agent uses to run experiments
agents/
  prompts.py           the system prompt for the coding agent
  coding_agent.py       the Gemini function-calling loop
baselines/
  nominal.py           baseline A, do nothing, just use the nominal model
  random_sysid.py      baseline B, random probing + scipy fit
scripts/
  sanity_check.py            checks the robot stands and walks without falling
  run_episode.py             runs baseline A and B against one hidden robot
  run_agent_episode.py       same thing, plus the Gemini agent
  run_benchmark.py           sweeps fault types x seeds x budgets, writes a CSV
  run_optimizer_comparison.py   fast fit vs slow-but-correct fit, same data
  render_snapshots.py        renders the pictures used in this README
  make_plots.py              turns a benchmark CSV into charts
report/                the write-up, with its own images
```

## setting it up

```
git submodule update --init --recursive
pip install -r requirements.txt
```

Check the robot actually works:

```
python scripts/sanity_check.py
```

You should see the standing controller hold at about 0.27m and the trot controller move forward without falling over. If it says "FELL", something's broken, probably the gait parameters are too aggressive again.

Run one full episode (secretly break something, run the random probing baseline, check if it beats doing nothing):

```
python scripts/run_episode.py
```

This takes a few minutes. It runs 8 experiments and fits 14 parameters with scipy, and each fit needs a bunch of full simulation rollouts, so just let it run.

To also run the coding agent, you need a Gemini API key. Export it, or drop a `.env` file in the repo root (it's gitignored, so it never gets committed):

```
GEMINI_API_KEY=your-key-here
```

Then:

```
python scripts/run_agent_episode.py
```

This makes real calls to the Gemini API, one per turn the agent takes, plus up to 8 more for the random baseline's fit. It prints the agent's tool calls as it goes so you can watch it think. If you want a different model, set `GEMINI_MODEL` (defaults to `gemini-3.5-flash`).

Full results and write-up: [`report/index.html`](report/index.html).
