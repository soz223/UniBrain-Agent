"""Run the pinned NEURA planning nodes; execute a selected UniBrain tool on the cluster."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PIN = "c349dde3e9d713181ecb7b734c8bb42fe9646c37"
TOOL = "unibrain_original_diagnosis"
QUESTION = (
    "Use the provided structural MRI dataset to predict individual ADHD versus control diagnosis. "
    "Use the supplied training and validation partitions. Fit only on training subjects, predict "
    "a continuous probability for every validation subject and report validation ROC AUC and accuracy. "
    "Keep the test partition closed. Use only available data; report any missing prerequisites."
)
SCHEMA = {
    "data_format": "NumPy structured arrays on a 96 x 96 x 96 grid",
    "needs_conversion": False,
    "modalities": ["anat"],
    "available_fields": ["subject identifier", "binary diagnosis", "T1w", "brain mask", "tissue mask", "AAL labels"],
    "partitions": ["train", "validation", "sealed test"],
    "geometry": "Original acquisition affines, spacing and DICOM headers are unavailable in the arrays.",
}


def registered_tool(args):
    from src.tools.registry import BaseTool, ExecutorType, Modality, ToolCallResult, ToolDefinition

    class UniBrainTool(BaseTool):
        @property
        def definition(self):
            return ToolDefinition(
                TOOL, "Train the original full UniBrain on the fixed structural MRI training fold and "
                "evaluate individual diagnosis on the validation fold. Takes the provided structured NumPy "
                "arrays directly and returns validation ROC AUC and accuracy.", "analysis", [Modality.ANAT],
                ExecutorType.PYTHON, {"type": "object", "properties": {}}, {"type": "object"})

        def execute(self, request):
            out = Path(request.output_dir)
            train = args.pipeline / "scripts/train_replacement.py"
            command = [str(args.python), str(train), "--source", str(args.source), "--data", str(args.fold),
                       "--output", str(out), "--dataset", "adhd", "--method", "original", "--setting", "original",
                       "--paper-original", "--fold", "1", "--epochs", str(args.epochs), "--patience", "50",
                       "--seed", "17", "--train-probe-size", "64", "--template-normalization", "minmax",
                       "--verify-gradients", "--checkpoint-mode", "resume", "--selection-metric", "classification_loss",
                       "--gpu", "0", "--threads", "2", "--coordination", "none", "--wall-seconds", str(args.wall_seconds)]
            if (out / "last.pt").is_file():
                command.extend(["--resume", str(out / "last.pt")])
            env = dict(os.environ, UNIBRAIN_SOURCES=str(args.source.parent))
            subprocess.run(command, check=True, cwd=args.pipeline, env=env)
            report = json.loads((out / "report.json").read_text())
            rows = report.get("history", [])
            if not rows:
                raise RuntimeError("No validation epoch in trainer report")
            return ToolCallResult(request.call_id, TOOL, "succeeded", outputs={
                "trainer_status": report["status"], "latest_validation": rows[-1], "report_path": str(out / "report.json")})

    return UniBrainTool()


def plan(args):
    from src.agent import graph_nodes as nodes
    from src.agent.graph_state import create_initial_state
    from src.agent.run_tracker import RunTracker
    from src.utils.llm import LLMClient

    class PlanningClient(LLMClient):
        def set_task_type(self, task_type):
            self.model = args.model

        def chat(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
            return super().chat(messages, temperature, min(max_tokens, 7500), **kwargs)

    nodes._llm = PlanningClient(api_base="https://api.openai.com/v1", api_key=os.environ["OPENAI_API_KEY"])
    nodes._llm.model = args.model
    nodes._tracker = RunTracker(str(args.out))
    nodes._get_data_structure = lambda: dict(SCHEMA)
    if args.public_kg:
        from src.knowledge.tool_knowledge_graph import TOOL_KNOWLEDGE_GRAPH
        for key in ("tools", "task_tool_mapping", "disease_roi_mapping", "tool_equivalences"):
            TOOL_KNOWLEDGE_GRAPH.setdefault(key, {})
    registry = nodes._get_registry()
    question = QUESTION
    if args.arm == "with_unibrain":
        registry.register(registered_tool(args))
        question += " Use the registered full UniBrain tool for training and validation."
    state = create_initial_state(args.arm, question)
    if args.revise:
        old = json.loads(args.revise.read_text(encoding="utf-8"))
        if old["arm"] != args.arm or old["question"] != question:
            raise ValueError("Review and current task differ")
        evidence = (
            "The previous plan failed its environment preflight. Available inputs are exactly the "
            "structured NumPy arrays described in the task. No DICOM, NIfTI, original affine or voxel "
            "spacing is provided; conversion, SPM and recon-all cannot run on these inputs. The local "
            "python_stats interface supports group tests, correlation and OLS; it does not implement "
            "subject-level classification. Use only executable steps needed for the requested output. "
            "If a prerequisite prevents the requested individual predictions, report it explicitly."
        )
        state["evidence"] = evidence + "\nPrior MoER review: " + json.dumps(old["review"], ensure_ascii=False)
        parsed = args.revise.parent / args.arm / "steps/01_parse_question/parsed_intent.json"
        state["parsed_intent"] = json.loads(parsed.read_text(encoding="utf-8"))
    for node in (nodes.node_init, nodes.node_parse_question, nodes.node_generate_plan):
        state.update(node(state))
        if state.get("last_error"):
            raise RuntimeError(state["last_error"])
    steps = state["plan"].get("pipeline", [])
    if not steps:
        raise RuntimeError("NEURA returned an empty pipeline")
    manifest = dict(arm=args.arm, upstream=PIN, model=args.model, question=question, schema=SCHEMA,
                    static_kg="empty public schema" if args.public_kg else "unchanged upstream",
                    nodes=["init", "parse_question", "generate_plan"], plan=state["plan"],
                    review=state.get("plan_review"), revised=bool(args.revise),
                    tool_names=[step.get("tool") for step in steps])
    (args.out / "decision.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def execute(args):
    from src.tools.registry import ToolCallRequest, ToolRegistry

    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    steps = [s for s in decision["plan"]["pipeline"] if s.get("tool") == TOOL]
    if decision["upstream"] != PIN or decision["arm"] != "with_unibrain" or len(steps) != 1:
        raise ValueError("Execution requires exactly one explicit registered UniBrain step from NEURA")
    registry = ToolRegistry()
    registry.register(registered_tool(args))
    result = registry.execute(ToolCallRequest(TOOL, "neura-unibrain", {}, {}, str(args.out / "training")))
    record = dict(selected_step=steps[0], scope="Selected UniBrain tool execution", status=result.status,
                  error=result.error, outputs=result.outputs, decision=decision)
    (args.out / "tool_execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if result.status != "succeeded":
        raise RuntimeError(result.error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neura", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    p = modes.add_parser("plan")
    p.add_argument("--arm", choices=["native", "with_unibrain"], required=True)
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--public-kg", action="store_true")
    p.add_argument("--revise", type=Path)
    e = modes.add_parser("execute")
    for key in ("decision", "pipeline", "source", "fold", "python"):
        e.add_argument("--" + key, type=Path, required=True)
    e.add_argument("--epochs", type=int, default=1000)
    e.add_argument("--wall-seconds", type=int, default=82500)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.neura.resolve()))
    (plan if args.mode == "plan" else execute)(args)
