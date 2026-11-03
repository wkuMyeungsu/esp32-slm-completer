// Python 파일에서 커서 앞 코드를 브리지(→ ESP32-S3)로 보내고, 받은 완성을 회색 인라인 제안으로 보여 준다.
import * as vscode from 'vscode';
import * as path from 'path';
import * as readline from 'readline';
import { spawn, ChildProcessWithoutNullStreams } from 'child_process';

let bridge: ChildProcessWithoutNullStreams | undefined;
const pending = new Map<number, (completion: string) => void>();
let nextId = 1;

export function activate(ctx: vscode.ExtensionContext) {
  const out = vscode.window.createOutputChannel('ESP32 SLM');
  // 설치본은 확장 폴더가 ~/.vscode/extensions로 옮겨지므로 설정의 bridge 경로를 쓴다. 비어 있으면 개발 모드(F5) 위치
  const bridgeDir = vscode.workspace.getConfiguration('esp32Slm').get<string>('bridgeDir')
    || path.join(ctx.extensionPath, '..', 'bridge');
  // Windows에서 uv가 .bat(pyenv-win shim)일 수 있어 셸로 실행한다. 명령은 고정 문자열
  bridge = spawn('uv run bridge.py', { cwd: bridgeDir, shell: true });
  bridge.on('error', (e) => {
    vscode.window.showWarningMessage(`ESP32 SLM: 브리지 실행 실패 (${e.message}), esp32Slm.bridgeDir 설정과 uv 설치를 확인하세요`);
    bridge = undefined;
  });
  bridge.stderr.on('data', (d) => out.append(d.toString()));
  bridge.on('exit', (code) => { out.appendLine(`bridge exited: ${code}`); bridge = undefined; });

  let markReady!: () => void;
  const ready = new Promise<void>((r) => (markReady = r));
  readline.createInterface({ input: bridge.stdout }).on('line', (line) => {
    out.appendLine(line);
    let msg: any;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.ready) { vscode.window.setStatusBarMessage(`ESP32 SLM: ${msg.port}`, 5000); markReady(); }
    else if (msg.id === undefined && msg.error) vscode.window.showWarningMessage(`ESP32 SLM: ${msg.error}`);
    pending.get(msg.id)?.(msg.completion ?? '');
    pending.delete(msg.id);
  });

  // 제안을 Tab으로 수락하면 VS Code가 이 명령을 실행한다 → 브리지가 accept 로그를 남김
  ctx.subscriptions.push(vscode.commands.registerCommand('esp32Slm.accepted', (id: number) => {
    bridge?.stdin.write(JSON.stringify({ type: 'accept', id }) + '\n');
  }));

  // shortcut: 키 입력마다 요청한다, 느려지면 디바운스 추가
  const provider: vscode.InlineCompletionItemProvider = {
    async provideInlineCompletionItems(doc, pos, _ctx, token) {
      if (!bridge) return;
      const prefix = doc.getText(new vscode.Range(new vscode.Position(0, 0), pos)).slice(-1024);
      const id = nextId++;
      const completion = await new Promise<string>((resolve) => {
        pending.set(id, resolve);
        bridge!.stdin.write(JSON.stringify({ id, prefix }) + '\n');
        setTimeout(() => { pending.delete(id); resolve(''); }, 3000);
      });
      if (!completion || token.isCancellationRequested) return;
      const item = new vscode.InlineCompletionItem(completion, new vscode.Range(pos, pos));
      item.command = { title: 'accepted', command: 'esp32Slm.accepted', arguments: [id] };
      return [item];
    },
  };
  ctx.subscriptions.push(vscode.languages.registerInlineCompletionItemProvider({ language: 'python' }, provider));
  return { provider, ready };  // 통합 테스트가 브리지 준비를 기다린 뒤 provider를 직접 호출한다
}

export function deactivate() {
  // kill은 셸만 죽이고 Python이 COM 포트를 쥔 채 남는다. stdin을 닫으면 브리지가 남은 로그를 보내고 스스로 끝난다
  bridge?.stdin.end();
}
