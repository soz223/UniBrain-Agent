import io
import json
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from unibrain_agent.agents import Agent
from unibrain_agent.providers import ProviderSettings, api_request, invoke, login_status, validate_profile
from unibrain_agent.server import create_server

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


class ProviderTests(unittest.TestCase):
    def test_settings_api_saves_preferences_and_blocks_changes_during_a_run(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = ProviderSettings(Path(folder) / 'settings.json')
            with patch('unibrain_agent.providers.provider_settings', return_value=settings):
                server = create_server(0)
            thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            def post(body, origin=None):
                headers = {'Content-Type': 'application/json'}
                if origin: headers['Origin'] = origin
                return urlopen(Request(base + '/api/ai/settings', data=json.dumps(body).encode(), headers=headers))
            try:
                body = {'provider': 'openai', 'model': 'test-model', 'api_key': 'secret-value'}
                with post(body) as response:
                    self.assertNotIn('secret-value', response.read().decode())
                with self.assertRaises(HTTPError) as blocked: post(body, 'https://untrusted.example')
                self.assertEqual(blocked.exception.code, 403)
                server.state['running'] = True
                with self.assertRaises(HTTPError) as blocked: post(body)
                self.assertEqual(blocked.exception.code, 409)
            finally:
                server.shutdown();server.server_close();thread.join()

    def test_connection_test_is_async_and_validates_boolean_output(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = ProviderSettings(Path(folder) / 'settings.json')
            with patch('unibrain_agent.providers.invoke', return_value=({'ok': 1}, None)):
                settings.start_test()
                deadline = time.monotonic() + 3
                while settings.public()['test']['status'] == 'running' and time.monotonic() < deadline:
                    time.sleep(.005)
            self.assertEqual(settings.public()['test']['status'], 'failed')

    def test_key_stays_in_memory_and_endpoint_change_clears_it(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {}, clear=True):
            path = Path(folder) / 'settings.json'
            settings = ProviderSettings(path)
            public = settings.update({"provider": "compatible", "model": "test", "base_url": "https://first.example/v1", "api_key": "secret-value"})
            self.assertNotIn('secret-value', json.dumps(public))
            self.assertNotIn('secret-value', path.read_text())
            self.assertEqual(settings.snapshot()['api_key'], 'secret-value')
            self.assertEqual(ProviderSettings(path).snapshot()['api_key'], '')
            settings.update({"provider": "compatible", "model": "test", "base_url": "https://second.example/v1"})
            self.assertEqual(settings.snapshot()['api_key'], '')

    def test_api_endpoint_and_model_validation(self):
        for address in ('http://remote.example/v1', 'file:///tmp/key', 'https://user:pass@example.com', 'https://example.com?key=value'):
            with self.assertRaises(ValueError):
                validate_profile('compatible', {"model": "test", "base_url": address})
        with self.assertRaises(ValueError): validate_profile('openai', {"model": ""})
        self.assertEqual(validate_profile('openai', {"model": "test", "base_url": "https://other.example"})['base_url'], 'https://api.openai.com/v1')

    def test_all_api_dialects_parse_output_and_use_correct_headers(self):
        payloads = {
            'openai': {"status": "completed", "output": [{"content": [{"type": "output_text", "text": '{"ok":true}'}]}]},
            'anthropic': {"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"ok":true}'}]},
            'compatible': {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]},
        }
        for provider, payload in payloads.items():
            with self.subTest(provider=provider):
                opener = Mock()
                opener.open.return_value = io.BytesIO(json.dumps(payload).encode())
                with patch('unibrain_agent.providers.build_opener', return_value=opener):
                    result, _ = api_request({"provider": provider, "model": "test-model", "api_key": "test-key", "base_url": "https://example.com/v1"}, 'Prompt', SCHEMA)
                request = opener.open.call_args.args[0]
                self.assertEqual(result, {"ok": True})
                body = json.loads(request.data)
                self.assertEqual(body['model'], 'test-model')
                if provider == 'anthropic':
                    self.assertEqual(request.get_header('X-api-key'), 'test-key')
                    self.assertEqual(body['output_config']['format']['schema'], SCHEMA)
                else: self.assertEqual(request.get_header('Authorization'), 'Bearer test-key')

    def test_redirect_cannot_forward_api_credentials(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(self.path)
                self.send_response(307)
                self.send_header('Location', '/unexpected')
                self.end_headers()
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError, 'HTTP 307'):
                api_request({"provider": "compatible", "model": "test", "api_key": "secret", "base_url": f"http://127.0.0.1:{server.server_port}/v1"}, 'Prompt', SCHEMA)
            self.assertEqual(received, ['/v1/chat/completions'])
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_http_error_body_cannot_disclose_a_key(self):
        opener = Mock()
        opener.open.side_effect = HTTPError('https://example.com', 401, 'secret-value', {}, io.BytesIO(b'secret-value'))
        with patch('unibrain_agent.providers.build_opener', return_value=opener):
            with self.assertRaises(RuntimeError) as caught:
                api_request({"provider": "openai", "model": "test", "api_key": "secret-value", "base_url": "https://example.com"}, 'Prompt', SCHEMA)
        self.assertNotIn('secret-value', str(caught.exception))

    def test_truncated_api_result_is_rejected(self):
        opener = Mock()
        opener.open.return_value = io.BytesIO(b'{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}')
        with patch('unibrain_agent.providers.build_opener', return_value=opener):
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                api_request({"provider": "compatible", "model": "test", "api_key": "key", "base_url": "https://example.com"}, 'Prompt', SCHEMA)

    def test_claude_bridge_disables_tools_and_parses_structured_output(self):
        process = subprocess.CompletedProcess([], 0, json.dumps({"structured_output": {"ok": True}, "usage": {"input_tokens": 10}}), '')
        with tempfile.TemporaryDirectory() as folder, patch('unibrain_agent.providers.shutil.which', return_value='claude'), patch('unibrain_agent.providers.subprocess.run', return_value=process) as execute:
            result, usage = invoke({"provider": "claude", "model": "chosen-model"}, 'Evidence', SCHEMA, Path(folder), 'review')
        command = execute.call_args.args[0]
        self.assertIn('--safe-mode', command)
        self.assertEqual(command[command.index('--tools') + 1], '')
        self.assertIn('--strict-mcp-config', command)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(usage['input_tokens'], 10)

    def test_login_status_returns_requested_identity_but_no_credentials(self):
        process = subprocess.CompletedProcess([], 0, '{"email":"private@example.com","accessToken":"never-return-this"}', '')
        with patch('unibrain_agent.providers.shutil.which', return_value='claude'), patch('unibrain_agent.providers.subprocess.run', return_value=process):
            status = login_status('claude')
        self.assertTrue(status['authenticated'])
        self.assertEqual(status['email'], 'private@example.com')
        self.assertNotIn('never-return-this', json.dumps(status))

    def test_agent_records_actual_provider_and_rejects_bad_model_output(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = ProviderSettings(Path(folder) / 'settings.json')
            settings.update({"provider": "anthropic", "model": "chosen-model", "api_key": "secret"})
            with patch('unibrain_agent.agents.provider_settings', return_value=settings), patch('unibrain_agent.agents.invoke', return_value=({"ok": True}, None)):
                agent = Agent('configured', Path(folder) / 'run', lambda *args, **kwargs: None)
                agent.decide('research', {}, SCHEMA, {})
            self.assertEqual(agent.calls[0]['backend'], 'anthropic')
            self.assertNotIn('secret', json.dumps(agent.calls))
            with patch('unibrain_agent.agents.invoke', return_value=({"invented": True}, None)):
                with self.assertRaises(ValueError): agent.decide('research', {}, SCHEMA, {})


if __name__ == '__main__':
    unittest.main()
