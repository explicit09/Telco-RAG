"""Development-only Codex CLI adapter. Not a held-out isolation boundary."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .generation import ANSWER_SCHEMA, make_payload, parse_answer, make_read_payload, parse_read_answer


class SubscriptionGenerator:
    def __init__(self, *, model=None, max_requests=3, timeout=180, executable='codex', development_only=True, deny_read_roots=(), allow_document_reads=False):
        if not development_only:
            raise ValueError('Subscription CLI is not certified for held-out evaluation')
        if max_requests < 1 or timeout <= 0:
            raise ValueError('positive limits required')
        self.allow_document_reads = allow_document_reads
        self.model, self.max_requests, self.timeout = model, max_requests, timeout
        self.executable, self.requests = executable, 0
        self.deny_read_roots = tuple(str(Path(root).resolve(strict=True)) for root in deny_read_roots)
        if self.deny_read_roots and not Path("/usr/bin/sandbox-exec").is_file():
            raise RuntimeError("OS read restrictions require macOS sandbox-exec")

    def generate(self, question, evidence):
        if self.requests >= self.max_requests:
            raise RuntimeError('Subscription request budget exhausted')
        payload_factory = make_read_payload if self.allow_document_reads else make_payload
        payload = payload_factory(question, evidence, model=self.model or 'configured-default', max_output_tokens=1200)
        prompt = '\n\n'.join(message['content'] for message in payload['input'])
        prompt += '\nReturn only the requested JSON object. Do not invoke tools or read local files.'
        settings = {
            'approval_policy': 'never', 'web_search': 'disabled', 'project_doc_max_bytes': 0,
            'skills.include_instructions': False, 'skills.bundled.enabled': False,
            'memories.use_memories': False, 'memories.generate_memories': False,
            'memories.dedicated_tools': False, 'agents.enabled': False,
            'tools.update_plan.enabled': False, 'tools.experimental_request_user_input.enabled': False,
        }
        for feature in ('shell_tool', 'unified_exec', 'view_image', 'js_repl', 'code_mode',
                        'code_mode_host', 'code_mode_only', 'apps', 'plugins', 'remote_plugin',
                        'recommended_plugins', 'skill_search', 'memories',
                        'external_agent_memory_import', 'multi_agent', 'multi_agent_v2',
                        'browser_use', 'in_app_browser', 'computer_use', 'image_generation',
                        'hooks', 'plugin_hooks', 'deferred_executor', 'request_permissions_tool',
                        'token_budget', 'current_time_reminder', 'goals', 'workspace_dependencies'):
            settings['features.' + feature] = False
        with tempfile.TemporaryDirectory(prefix='raglab-dev-') as folder:
            folder = Path(folder)
            schema, output = folder / 'schema.json', folder / 'answer.json'
            schema.write_text(json.dumps(payload['text']['format']['schema']))
            command = [self.executable, 'exec', '--ignore-user-config', '--ephemeral', '--json',
                       '--sandbox', 'read-only', '--skip-git-repo-check', '-C', str(folder),
                       '--output-schema', str(schema), '--output-last-message', str(output)]
            if self.model:
                command += ['--model', self.model]
            for key, value in settings.items():
                command += ['-c', key + '=' + json.dumps(value)]
            command += ['-']
            if self.deny_read_roots:
                restrictions = ' '.join('(deny file-read* file-write* (subpath ' + json.dumps(root) + '))' for root in self.deny_read_roots)
                profile = '(version 1) (allow default) ' + restrictions
                command = ['/usr/bin/sandbox-exec', '-p', profile, *command]
            environment = dict(os.environ)
            for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_ORG_ID', 'OPENAI_PROJECT_ID'):
                environment.pop(key, None)
            self.requests += 1
            completed = subprocess.run(command, input=prompt, text=True, capture_output=True,
                                       cwd=folder, env=environment, timeout=self.timeout)
            if completed.returncode:
                # Avoid printing stderr, which may contain local configuration details.
                raise RuntimeError(f'Codex CLI exited {completed.returncode}; no prediction recorded')
            usage = {}
            startup_warnings = []
            for line in completed.stdout.splitlines():
                event = json.loads(line)
                kind = event.get('type')
                if kind in ('item.started', 'item.updated', 'item.completed'):
                    item = event.get('item', {})
                    if item.get('type') == 'error' and item.get('message') == 'Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`.':
                        startup_warnings.append(item['message'])
                        continue
                    if event.get('item', {}).get('type') not in ('agent_message', 'reasoning'):
                        raise RuntimeError('Non-answer event detected; reject development run: ' + str(event.get('item', {}).get('type')) + '; ' + str(event.get('item', {}).get('message', ''))[:300])
                elif kind == 'turn.completed':
                    usage = event.get('usage', {})
                elif kind not in ('thread.started', 'turn.started'):
                    raise RuntimeError('Unexpected CLI event; reject this development run')
            if not output.is_file():
                raise RuntimeError('CLI returned no structured answer')
            parser = parse_read_answer if self.allow_document_reads else parse_answer
            return parser(question, evidence, json.loads(output.read_text()),
                                {'provider': 'codex_subscription', 'development_only': True,
                                 'model': self.model or 'configured-default', 'usage': usage,
                                 'request': self.requests, 'startup_warnings': startup_warnings, 'os_denied_roots': self.deny_read_roots})
