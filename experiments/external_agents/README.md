# External agent experiments

The diagnosis task uses fold 1 of ADHD (ADHD versus control) and ABIDE (autism versus control). Preserve train and validation subject membership. Supply the agent with a task and tensor schema, freeze its implementation, and evaluate the implementation locally on subject images. The held-out test partition stays closed. Report AUROC, accuracy, sensitivity and specificity per dataset. A successful conversion or tool call is a separate outcome.

For system comparisons, retain upstream revision, model version, tool permissions, human adaptations, execution failures and generation budget. The frozen classifiers below each came from a single generation. Cross-validation over the subjects cannot measure variability across agent generations.

## NEXUS

The supervisor dispatched the downstream modeling agent, which retrieved a T1 MRI guideline and generated a classifier. The adaptation exposed the file writer to the downstream agent and used a separate evaluator. See the [generation instructions and dependency lock](nexus/README.md). To reproduce the validation run from this directory:

```sh
python nexus/evaluate.py "$FOLD_DIR" nexus/generation_01 "$NEW_RESULT_DIR" --threshold 0.5
```

## Codex

The coding agent received [the task](codex/task.txt) and implemented the [frozen classifier](codex/model.py) in an empty workspace. It uses two resolutions of local mean MRI intensity and ridge regression. The offline evaluator supplies MRI arrays and training labels, checks subject separation, verifies the frozen file hash, and scores validation predictions.

```sh
python ../../scripts/evaluate_external_classifier.py codex/model.py 53c19e384a3b65b83bed79ef0e8a08d32ae3fa11627706bfdc5ae5b60ae1459d "$FOLD_DIR" "$NEW_RESULT_DIR"
```

## OpenClaw with original source

OpenClaw 2026.11.3 ran its isolated coding loop with GPT 5.2 on the [task](openclaw/task.txt) and four public files from [the original implementation](https://github.com/Anonymous7852/UniBrain/tree/f9f8d906b1725046d532f862a28876482aca2cfa). It generated the [adapter](openclaw/adapter.py) in nine assistant turns with eight tool calls. On real ADHD subjects, two Adam updates produced finite gradients and parameter changes. Four held-out validation subjects returned finite probabilities. This protocol checks integration and optimizer updates, with insufficient data for a diagnostic performance estimate.

```sh
python openclaw/adapter.py --source "$ORIGINAL_SRC" --fold "$ADHD_FOLD" --output "$NEW_RESULT_DIR" --steps 2
```

## NeuroPilot

The publicly available [DICOM conversion skill](https://github.com/acmlab/dcm2bids-skill/tree/7e9880588e203404d08652aee576a2c68ee71f87) takes acquisition DICOM files. An agent inspected metadata from the author's synthetic fixture and produced the saved conversion rules. The replay checks series assignment, six image and sidecar outputs, and BIDS validation. The resulting measurements concern data standardization.

Use dcm2bids 3.3.1, dcm2niix 1.0.20261029, pydicom 3.0.2 and the Node bids-validator 1.15.0 CLI. Clone the linked skill at its pinned revision first.

```sh
python "$SKILL_DIR/tests/generate_fixture.py"
python neuropilot/replay_bids_skill.py "$SKILL_DIR/tests/fixtures/synthetic/sub01" neuropilot/dcm2bids_config.json "$NEW_RESULT_DIR" "$VALIDATOR_JS"
```

## NEURA

Use the [public source](https://github.com/NeuroScienceLab/NEURA/tree/c349dde3e9d713181ecb7b734c8bb42fe9646c37). The unmodified planner raised `KeyError: task_tool_mapping`. The authors omit the static knowledge graph. The public adaptation supplies empty mappings, retains the available dynamic knowledge, and routes the planner and reviewer to GPT 4.1 mini. Both receive the task and input schema.

```sh
python neura/neura_arm.py --neura "$NEURA_SRC" --out "$NEW_PLAN_DIR" plan --arm native --public-kg
python neura/neura_arm.py --neura "$NEURA_SRC" --out "$NEW_TOOL_PLAN" plan --arm with_unibrain --public-kg
python neura/neura_arm.py --neura "$NEURA_SRC" --out "$REVISED_TOOL_PLAN" plan --arm with_unibrain --public-kg --revise "$NEW_TOOL_PLAN/decision.json"
```

The [four planning outcomes](neura/plan_audit.json) include initial and revised plans. The native arm proposed a classification operation absent from its statistics tool. On synthetic inputs the tool reported `succeeded`, but its result contained `Unknown analysis type: classification` and no predictions. In the registered-tool arm the revised plan put the supplied original UniBrain trainer first and proposed an additional statistics step. The reviewer still requested revision. The cluster experiment exercises that selected training tool for one epoch on ADHD fold 1. Measure tool execution separately from acceptance of the full plan.

```sh
python neura/neura_arm.py --neura "$NEURA_SRC" --out "$NEW_RESULT_DIR" execute --decision "$REVISED_TOOL_PLAN/decision.json" --pipeline "$AGENT_REPO" --source "$ORIGINAL_SRC" --fold "$ADHD_FOLD" --python "$TRAIN_PYTHON" --epochs 1
```
