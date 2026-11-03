import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';

const FIXTURE = path.resolve(__dirname, '../../../test-fixture');  // 가짜 브리지
const ACCEPT_LOG = process.env.FAKE_ACCEPT_LOG!;

suite('esp32-slm-completer', () => {
  test('Python 파일에서 제안을 받고, 수락하면 브리지에 accept가 간다', async () => {
    fs.rmSync(ACCEPT_LOG, { force: true });
    await vscode.workspace.getConfiguration('esp32Slm')
      .update('bridgeDir', FIXTURE, vscode.ConfigurationTarget.Global);

    const doc = await vscode.workspace.openTextDocument({ language: 'python', content: 'print(' });
    const ext = vscode.extensions.getExtension('wkuMyeungsu.esp32-slm-completer');
    assert.ok(ext, '확장이 설치되어 있어야 한다');
    const api = await ext.activate();
    await api.ready;  // uv run 첫 실행은 수 초 걸린다

    const items = await api.provider.provideInlineCompletionItems(
      doc, doc.positionAt(6), {} as vscode.InlineCompletionContext,
      new vscode.CancellationTokenSource().token) as vscode.InlineCompletionItem[];
    assert.strictEqual(items?.[0]?.insertText, '  # fake');

    const cmd = items[0].command!;
    await vscode.commands.executeCommand(cmd.command, ...(cmd.arguments ?? []));
    // 파일이 생긴 뒤 내용이 쓰이기까지 틈이 있어서, 내용이 들어올 때까지 기다린다
    const read = () => (fs.existsSync(ACCEPT_LOG) ? fs.readFileSync(ACCEPT_LOG, 'utf-8').trim() : '');
    for (let i = 0; i < 50 && !read(); i++) await new Promise((r) => setTimeout(r, 100));
    assert.strictEqual(read(), String(cmd.arguments![0]));
  });
});
