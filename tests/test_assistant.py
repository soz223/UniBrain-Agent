import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

from unibrain_agent.assistant_workspace import AssistantWorkspace, library_sources, latest_context, research_result, search_publications
from unibrain_agent.roles import run_team, measured_tasks
from unibrain_agent.server import create_server


class AssistantTests(unittest.TestCase):
    def complete(self, workspace, request):
        identifier = workspace.submit(request)["id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            record = workspace.read(identifier)
            if record["status"] != "running":
                while workspace.active and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertEqual(record["status"], "complete", record["error"])
                return record
            time.sleep(.005)
        self.fail("Assistant job did not finish")

    def test_conversation_is_persisted_with_followup_context(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = AssistantWorkspace(folder, backend="rules")
            first = self.complete(workspace, {"mode":"research", "message":"Compare registration", "stage":"registration", "online":False})
            second = self.complete(workspace, {"mode":"research", "message":"Which is cheaper?", "previous":first["id"], "stage":"registration", "online":False})
            self.assertEqual(second["history"][0]["question"], first["message"])
            self.assertTrue(second["history"][0]["sources"])
            self.assertEqual(AssistantWorkspace(folder).read(second["id"])["result"], second["result"])

    def test_online_failure_is_disclosed_and_does_not_fake_retrieval(self):
        with tempfile.TemporaryDirectory() as folder, patch("unibrain_agent.assistant_workspace.search_publications", side_effect=OSError("Unavailable")):
            record = self.complete(AssistantWorkspace(folder,"rules"), {"mode":"research","message":"Registration", "stage":"registration"})
            self.assertEqual(record["search"]["status"], "failed")
            self.assertTrue(any(event["kind"] == "warning" for event in record["events"]))
            self.assertFalse(any(source["source_id"].startswith("X") for source in record["sources"]))

    def test_unknown_citations_and_wrong_stage_tools_are_rejected(self):
        sources = library_sources("registration")
        result = {"answer":"Comparison", "source_ids":["M06"], "recommendations":[{"source_id":"M06", "stage":"registration", "tool_id":"M06", "reason":"Available core", "verification_needed":"Gradient check"}], "next_question":""}
        self.assertEqual(research_result(result,sources),result)
        result["recommendations"][0]["stage"] = "classification"
        with self.assertRaises(ValueError): research_result(result,sources)
        result["source_ids"] = ["invented"]
        with self.assertRaises(ValueError): research_result(result,sources)

    def test_role_team_requires_measured_gradients_before_balancing(self):
        class Recorder:
            def __init__(self): self.calls=[]
            def decide(self,role,context,schema,fallback,instructions=None):
                self.calls.append((role,context,instructions))
                return {"action":"balance_gradients","rationale":"Requested by test"} if role == "coordinator" else fallback
        agent=Recorder()
        result=run_team(agent,{"diagnostic":None,"observed_module_updates":["ext_net"]})
        self.assertEqual(result["action"],"request_measurement")
        self.assertEqual([call[0] for call in agent.calls],["extraction","registration","segmentation","classification","coordinator"])
        self.assertEqual(len(agent.calls[-1][1]["specialists"]),4)
        self.assertFalse(measured_tasks({"diagnostic":{"task_norms":{"extraction":float('nan')}}}))

    def test_review_context_excludes_subject_records_and_local_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); run=root/'case'; run.mkdir()
            (root/'latest.json').write_text(json.dumps({"run_dir":str(run.resolve())}))
            (run/'report.json').write_text(json.dumps({"candidate":{"history":[{"subject_id":"private-subject"}], "validation":{"count":5,"records":[{"path":"private-path"}]}},"data":{"ids":["private-subject"]}}))
            with patch("unibrain_agent.assistant_workspace.RUNS",root):
                context=latest_context()
            self.assertNotIn("private",json.dumps(context))
            self.assertEqual(context["validation"],{"count":5})

    def test_live_metadata_remains_an_unverified_literature_lead(self):
        payload={"message":{"items":[{"DOI":"10.1/example","title":["A <b>method</b>"],"published":{"date-parts":[[2024]]},"container-title":["Journal"],"abstract":"<jats:p>Measured abstract.</jats:p>"}]}}
        with patch("unibrain_agent.assistant_workspace.urlopen",return_value=io.BytesIO(json.dumps(payload).encode())):
            sources,query=search_publications("registration","VoxelMorph")
        self.assertIn("Registration",query)
        self.assertEqual(sources[0]["tool_id"],"")
        self.assertIn("not checked",sources[0]["verification"])
        self.assertNotIn("<",sources[0]["abstract"])

    def test_async_http_endpoint_returns_a_saved_result(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace=AssistantWorkspace(folder,"rules")
            with patch("unibrain_agent.assistant_workspace.AssistantWorkspace",return_value=workspace):
                server=create_server(0)
            thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
            base=f"http://127.0.0.1:{server.server_port}"
            try:
                request=Request(base+'/api/assistant',data=json.dumps({"mode":"research","message":"Explain extraction","online":False}).encode(),headers={"Content-Type":"application/json"})
                with urlopen(request) as response:
                    self.assertEqual(response.status,202)
                    identifier=json.load(response)["id"]
                deadline=time.monotonic()+3
                while workspace.active and time.monotonic()<deadline: time.sleep(.005)
                with urlopen(base+'/api/assistant/jobs/'+identifier) as response: result=json.load(response)
                self.assertEqual(result["status"],"complete")
                self.assertEqual(result["backend"],"rules")
            finally:
                server.shutdown();server.server_close();thread.join()

    def test_invalid_request_cannot_start_a_job(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace=AssistantWorkspace(folder,"rules")
            for body in ({"mode":"research","message":"x","stage":[]},{"mode":"roles","role":"shell"},{"mode":"research","message":"x","previous":"../outside"},{"mode":"research","message":"x","selection":[]}):
                with self.assertRaises(ValueError): workspace.submit(body)
            self.assertIsNone(workspace.active)


if __name__ == "__main__":
    unittest.main()
