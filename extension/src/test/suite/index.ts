import * as path from 'path';
import Mocha from 'mocha';

export function run(): Promise<void> {
  const mocha = new Mocha({ ui: 'tdd', timeout: 60000 });
  mocha.addFile(path.resolve(__dirname, 'extension.test.js'));
  return new Promise((resolve, reject) =>
    mocha.run((failures) => (failures ? reject(new Error(`${failures} test(s) failed`)) : resolve())));
}
