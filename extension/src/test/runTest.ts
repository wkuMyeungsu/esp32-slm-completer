// VS Code를 내려받아 확장을 개발 모드로 띄우고 suite를 실행한다.
import * as os from 'os';
import * as path from 'path';
import { runTests } from '@vscode/test-electron';

runTests({
  vscodeExecutablePath: process.env.VSCODE_EXECUTABLE_PATH,  // 비우면 테스트용 VS Code를 내려받는다 (CI)
  extensionDevelopmentPath: path.resolve(__dirname, '../../'),
  extensionTestsPath: path.resolve(__dirname, './suite/index'),
  launchArgs: ['--disable-extensions'],
  extensionTestsEnv: { FAKE_ACCEPT_LOG: path.join(os.tmpdir(), 'esp32-slm-accept.txt') },
}).catch((e) => {
  console.error(e);
  process.exit(1);
});
