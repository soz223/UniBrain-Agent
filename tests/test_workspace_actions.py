import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from unibrain_agent.access import AccessPolicy
from unibrain_agent.builds import create_build, generated_code, review_code
from unibrain_agent.catalog import ADAPTER_DEFAULTS
from unibrain_agent.data import digest
from unibrain_agent.paths import PROJECT
from unibrain_agent.providers import ProviderSettings
from unibrain_agent.server import create_server


class WorkspaceActionTests(unittest.TestCase):
    def test_ai_requires_manual_connection_and_disconnect_revokes_the_session(self):
        with tempfile.TemporaryDirectory() as folder:
            settings=ProviderSettings(Path(folder)/'ai.json')
            with patch('unibrain_agent.providers.provider_settings',return_value=settings), patch('unibrain_agent.accounts.login_status',return_value={'authenticated':True,'email':'test@example.com'}):
                server=create_server(0,access=AccessPolicy(token='',origins=[]))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            def request(path,body=None,session=None):
                headers={'Content-Type':'application/json'}
                if session:headers['X-UniBrain-AI-Session']=session
                return urlopen(Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers=headers),timeout=5)
            try:
                for route,body in [('/api/ai/login?provider=codex',None),('/api/ai/test',{}),('/api/assistant',{'mode':'research','message':'test'})]:
                    with self.assertRaises(HTTPError) as caught:request(route,body)
                    self.assertEqual(caught.exception.code,403)
                with request('/api/ai/connect',{'confirm':True}) as response:grant=json.load(response)
                self.assertEqual(grant['account']['email'],'test@example.com')
                with request('/api/ai/login?provider=codex',session=grant['session']) as response:self.assertTrue(json.load(response)['authenticated'])
                with request('/api/ai/disconnect',{},grant['session']):pass
                with self.assertRaises(HTTPError):request('/api/ai/login?provider=codex',session=grant['session'])
            finally:
                server.shutdown();server.server_close();thread.join()

    def test_http_budget_accepts_1000_and_rejects_larger_values_without_starting_training(self):
        server=create_server(0,access=AccessPolicy(token='',origins=[]));server.state['running']=True
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            for steps,expected in [(500,409),(1000,409),(1001,400),(1,400),(True,400)]:
                request=Request(f'http://127.0.0.1:{server.server_port}/api/run',data=json.dumps({'backend':'rules','steps':steps}).encode(),headers={'Content-Type':'application/json'})
                with self.assertRaises(HTTPError) as caught:urlopen(request)
                self.assertEqual(caught.exception.code,expected)
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_review_contains_execution_code_and_detects_a_modified_entry_point(self):
        with tempfile.TemporaryDirectory() as folder,patch('unibrain_agent.builds.BUILDS',Path(folder)):
            record=create_build(ADAPTER_DEFAULTS,'rules')
            root=Path(folder)/record['id'];code=generated_code(ADAPTER_DEFAULTS)
            (root/'pipeline.py').write_text(code)
            record.update(code=code,code_sha256=digest(root/'pipeline.py'),diff='reviewable diff',implementation_hashes={str(p.relative_to(PROJECT)):digest(p) for p in (PROJECT/'unibrain_agent').glob('*.py')})
            (root/'build.json').write_text(json.dumps(record))
            result=review_code(record['id'])
            self.assertEqual(result['files'][0]['code'],code)
            self.assertIn('unibrain_agent/composition.py',[item['name'] for item in result['files']])
            (root/'pipeline.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):review_code(record['id'])


if __name__ == '__main__':
    unittest.main()
