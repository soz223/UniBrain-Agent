"""Run the pinned NEXUS supervisor and downstream agent on an input schema."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


REVISION = "c2e15d4983489fd89d604fdbbefbbf1726958068"


def generate(source, workspace):
    import httpx
    import langchain_openai
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("Unexpected upstream revision")
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    program = workspace / "diagnosis.py"
    calls = []
    lock = threading.Lock()

    def limit_requests(request):
        with lock:
            if len(calls) >= 12:
                raise RuntimeError("Experiment model call limit reached")
            calls.append(request.url.path)

    original_chat = langchain_openai.ChatOpenAI
    client = httpx.Client(event_hooks={"request": [limit_requests]}, timeout=90)

    def bounded_chat(**kwargs):
        return original_chat(**kwargs, max_retries=0, max_tokens=4800, timeout=90, http_client=client)

    langchain_openai.ChatOpenAI = bounded_chat
    sys.path.insert(0, str(source.resolve() / "source"))
    import downstream_analysis_agent as downstream
    import master_control_agent as master

    write = next(t for t in downstream.downstream_analysis_tools if t.name == "write_file")

    @tool("write_file")
    def write_program(file_path: str, content: str) -> str:
        """Write diagnosis.py to the single absolute path supplied by the supervisor."""
        if Path(file_path) != program or program.is_symlink():
            raise ValueError(f"Use {program}")
        return write.invoke({"file_path": str(program), "content": content})

    downstream.downstream_analysis_tools = [write_program]
    graph = create_react_agent(model=master.llm, tools=[master.call_downstream_analysis_agent],
                               prompt=master.SUPERVISOR_SYSTEM_PROMPT, state_schema=master.SupervisorState)
    request = f"""Develop the NEXUS downstream sMRI disease classifier for ADHD versus control. The same
    method will be applied independently to ABIDE ASD versus control. The inputs are prepared whole-head
    T1 MRI arrays on a common 96x96x96 array grid. No affines, tissue probabilities, atlas maps, voxel
    spacing, demographics, external labels, or functional time series are available to the classifier.
    Use the Downstream Analysis Agent and its original resource retrieval to choose a method appropriate
    for this data. Preserve differences between these inputs and the MNI gray matter probability guide.
    This invocation generates the program only. A separate offline Delta worker executes the reviewed
    program and computes metrics. There is no dataset directory for you to inspect.
    Ask the agent to call write_file once with the absolute destination {program}.
    Deliver a pure Python module defining fit_predict(x_train, y_train, x_eval), returning exactly one
    finite continuous positive-class score per evaluation subject. x_train and x_eval are read-only
    float32 numpy arrays with shape (N,96,96,96), possibly strided or memory-mapped. y_train is a 1D
    integer 0/1 array. The preserved training fold has about 465 to 651 subjects. No validation or test
    labels are accessible to the model. Fit preprocessing only on training, transform eval subjects
    independently. You may use numpy, scipy and sklearn. Use deterministic randomness, one CPU thread
    and under 8GB RAM. Iterate over subjects for feature extraction; no train-size dense volume copy.
    Choose an effective structural representation and regularized predictive model. Any hyperparameter
    search must use training-only folds and be practical with a 30 minute CPU budget. No filesystem I/O,
    subprocess, networking, installation, external model downloads, or executing shell commands inside
    generated code. No mock results. Report when the code has been written for external testing."""
    state = {name: [] for name in master.SupervisorState.__annotations__ if name != "user_request"}
    state.update(user_request=request, messages=[HumanMessage(content=request)])
    state_result = graph.invoke(state, config={"recursion_limit": 10})
    if not program.is_file():
        raise RuntimeError("NEXUS did not write the classifier")
    result = {"upstream": revision, "llm": "gpt-5.2", "request_count": len(calls),
              "model_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
              "retrieved_tools": state_result.get("downstream_analysis_agent_selected_tools", []),
              "retrieved_knowledge": state_result.get("downstream_analysis_agent_selected_knowledge_docs", []),
              "supervisor_dispatches": len(state_result.get("supervisor_to_downstream_analysis_instructions", [])),
              "downstream_reports": len(state_result.get("downstream_analysis_to_supervisor_msgs", [])),
              "scope": "Upstream supervisor, downstream retrieval and ReAct generation on schema only"}
    if not result["supervisor_dispatches"] or not result["downstream_reports"]:
        raise RuntimeError("Missing native inter-agent messages")
    (workspace / "generation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    generate(args.source, args.workspace)
