import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from unibrain_agent.access import AccessPolicy
from unibrain_agent.accounts import login_status, logout
from unibrain_agent.frontend import document
from unibrain_agent.server import create_server

TOKEN = 'test-access-token-with-at-least-32-characters'
ORIGIN = 'https://soz223.github.io'


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.server = create_server(0, access=AccessPolicy(token=TOKEN, origins=[ORIGIN]))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def request(self, path='/api/status', token=TOKEN, origin=ORIGIN, body=None, method=None):
        headers = {'Origin': origin, 'Content-Type': 'application/json'}
        if token: headers['Authorization'] = 'Bearer ' + token
        return urlopen(Request(self.base + path, data=json.dumps(body).encode() if body is not None else None, headers=headers, method=method))

    def test_cross_origin_api_requires_token_and_exact_origin(self):
        with self.request() as response:
            self.assertEqual(response.headers['Access-Control-Allow-Origin'], ORIGIN)
            self.assertEqual(json.load(response)['api_version'], 2)
        for token in (None, 'incorrect'):
            with self.assertRaises(HTTPError) as caught: self.request(token=token)
            self.assertEqual(caught.exception.code, 401)
        with self.assertRaises(HTTPError) as caught: self.request(origin='https://untrusted.example')
        self.assertEqual(caught.exception.code, 403)
        self.assertIsNone(caught.exception.headers.get('Access-Control-Allow-Origin'))

    def test_objective_preview_is_authenticated_and_returns_backend_terms(self):
        from unibrain_agent.catalog import DEFAULTS
        body = {'selection': {**DEFAULTS, 'denoising': 'M14', 'registration': 'M06'}}
        with self.request('/api/objective/preview', body=body) as response:
            result = json.load(response)
        self.assertEqual(result['status'], 'declared')
        self.assertIn('regularization', [term['id'] for term in result['terms']])
        self.assertIn('denoising', [module['id'] for module in result['modules']])
        with self.assertRaises(HTTPError):
            self.request('/api/objective/preview', token=None, body=body)

    def test_training_monitor_uses_existing_auth_and_registered_ids(self):
        for path in ('/api/training-runs', '/api/training-runs/example'):
            with self.assertRaises(HTTPError) as caught:
                self.request(path, token=None)
            self.assertEqual(caught.exception.code, 401)
        with patch('unibrain_agent.training_monitor.TrainingMonitor.listing', return_value=[{'id':'example','label':'Example'}]):
            with self.request('/api/training-runs') as response:
                self.assertEqual(json.load(response)['runs'][0]['id'], 'example')
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/training-runs/unregistered')
        self.assertEqual(caught.exception.code, 404)

    def test_preflight_is_available_without_credentials_but_data_is_not(self):
        with self.request(token=None, method='OPTIONS') as response:
            self.assertIn('Authorization', response.headers['Access-Control-Allow-Headers'])
        for path in ('/api/ai', '/api/ai/login?provider=codex', '/api/report', '/api/assistant'):
            with self.assertRaises(HTTPError) as caught: self.request(path, token=None)
            self.assertEqual(caught.exception.code, 401)

    def test_public_shell_contains_no_saved_private_report(self):
        with self.request('/', token=None) as response:
            html = response.read().decode()
        self.assertIn('id="reportData" type="application/json">null</script>', html)
        self.assertNotIn(TOKEN, html)

    def test_logout_needs_explicit_confirmation_and_checks_active_jobs(self):
        with patch('unibrain_agent.accounts.logout', return_value={'authenticated': False}) as signout:
            with self.assertRaises(HTTPError) as caught:
                self.request('/api/ai/logout', body={'provider': 'codex'})
            self.assertEqual(caught.exception.code, 400)
            signout.assert_not_called()
            self.server.state['running'] = True
            with self.assertRaises(HTTPError) as caught:
                self.request('/api/ai/logout', body={'provider': 'codex', 'confirm': True})
            self.assertEqual(caught.exception.code, 409)

    def test_report_endpoint_loads_only_the_configured_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'sample';run.mkdir()
            (root/'latest.json').write_text(json.dumps({'run_dir':str(run.resolve())}))
            (run/'report.json').write_text(json.dumps({'status':'complete','run_dir':str(run)}))
            with patch('unibrain_agent.server.RUNS',root):
                with self.request('/api/report') as response: result=json.load(response)
            self.assertEqual(result['report']['status'],'complete')
            self.assertEqual(result['report']['previews'],{})


class AccountAndStaticTests(unittest.TestCase):
    def test_codex_account_returns_identity_without_reading_raw_credentials(self):
        with patch('unibrain_agent.accounts.shutil.which',return_value='codex'), patch('unibrain_agent.accounts.codex_account',return_value={'authenticated':True,'email':'user@example.com','plan':'pro','method':'chatgpt'}):
            state=login_status('codex')
        self.assertEqual(state['email'],'user@example.com')
        self.assertEqual(state['plan'],'pro')
        self.assertTrue(state['authenticated'])

    def test_logout_calls_only_the_selected_cli(self):
        with patch('unibrain_agent.accounts.shutil.which',return_value='codex'), patch('unibrain_agent.accounts.subprocess.run') as execute, patch('unibrain_agent.accounts.login_status',return_value={'authenticated':False}):
            execute.return_value.returncode=0
            self.assertFalse(logout('codex')['authenticated'])
            self.assertEqual(execute.call_args.args[0],['codex','logout'])

    def test_remote_policy_rejects_weak_tokens_and_insecure_public_origins(self):
        with self.assertRaises(ValueError):AccessPolicy(host='0.0.0.0',token='')
        with self.assertRaises(ValueError):AccessPolicy(token='short',origins=[ORIGIN])
        with self.assertRaises(ValueError):AccessPolicy(token=TOKEN,origins=['http://gpu.example'])
        policy=AccessPolicy(token=TOKEN,public_url='https://gpu.example')
        self.assertFalse(policy.accepts_origin('http://gpu.example',8776))
        self.assertTrue(policy.accepts_origin('https://gpu.example',8776))

    def test_static_build_contains_public_catalog_logos_and_no_private_payload(self):
        html=document(standalone=True)
        self.assertIn('"standalone": true',html)
        self.assertIn('VoxelMorph',html)
        self.assertIn('data-provider="claude"',html)
        self.assertNotIn('__LOGO_',html)
        self.assertNotIn('__CONNECTION_SCRIPT__',html)
        self.assertNotIn('__OBJECTIVE_PANEL__',html)
        self.assertNotIn('__TRAINING_PANEL__',html)
        self.assertIn('id="trainingLab"',html)
        self.assertIn('id="objectivePanel"',html)
        self.assertIn('function objectiveFormula(',html)
        self.assertNotIn('C:\\Users\\',html)
        self.assertIn('id="reportData" type="application/json">null</script>',html)


if __name__ == '__main__':
    unittest.main()
