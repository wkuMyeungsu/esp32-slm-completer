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
  bridge = spawn('uv', ['run', 'bridge.py'], { cwd: path.join(ctx.extensionPath, '..', 'bridge') });
  bridge.stderr.on('data', (d) => out.append(d.toString()));
  bridge.on('exit', (code) => { out.appendLine(`bridge exited: ${code}`); bridge = undefined; });

  readline.createInterface({ input: bridge.stdout }).on('line', (line) => {
    out.appendLine(line);
    let msg: any;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.ready) vscode.window.setStatusBarMessage(`ESP32 SLM: ${msg.port}`, 5000);
    else if (msg.id === undefined && msg.error) vscode.window.showWarningMessage(`ESP32 SLM: ${msg.error}`);
    pending.get(msg.id)?.(msg.completion ?? '');
    pending.delete(msg.id);
  });

  // 제안을 Tab으로 수락하면 VS Code가 이 명령을 실행한다 → 브리지가 accept 로그를 남김
  ctx.subscriptions.push(vscode.commands.registerCommand('esp32Slm.accepted', (id: number) => {
    bridge?.stdin.write(JSON.stringify({ type: 'accept', id }) + '\n');
  }));

  // shortcut: 키 입력마다 요청한다, 느려지면 디바운스 추가
  ctx.subscriptions.push(vscode.languages.registerInlineCompletionItemProvider({ language: 'python' }, {
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
  }));
}

export function deactivate() {
  bridge?.kill();
}
