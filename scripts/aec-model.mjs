#!/usr/bin/env node
// Standalone maintainer entry for this repository: build, verify and Draft-publish DTLN-aec ONNX artifacts.
// No application source lives here; the VoiceInsight app only consumes published releases.
import { mkdir, writeFile, readFile, rm, access } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { runProcess, safeEnv, requireThat, validateVersion, validateVariant, selectSpec, validateRepo, validateTarget, json, toolHashes, inspectBundle, publishDraft, githubClient } from '../tools/aec-model/release.mjs';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const tool = join(project, 'tools/aec-model');
const usage = `DTLN-aec ONNX artifact maintenance (this repository only)
  npm run model:aec:build -- --variant 128|256|512 --version 1.0.0
  npm run model:aec:verify -- --variant 128|256|512 --version 1.0.0 --directory <bundle>
  npm run model:aec:publish -- --variant 128|256|512 --version 1.0.0 --directory <bundle> --repo owner/repo --target <40-hex-commit> --dry-run
  npm run model:aec:publish -- --variant 128|256|512 --version 1.0.0 --directory <bundle> --repo owner/repo --target <40-hex-commit> --confirm-draft --acknowledge-limitations
Build requires uv 0.12.22 and macOS arm64; installs a locked private Python environment.
Verify/publish require that environment, never download/convert models. Dry-run has ZERO GitHub calls.
Publish creates a NEW DRAFT ONLY, checks remote asset hashes, never overwrites or makes public.`;

function parse(args) {
  const [action, ...rest] = args;
  if (rest.includes('--help') || action === '--help') return { help: true };
  requireThat(['build', 'verify', 'publish'].includes(action), 'choose build, verify or publish');
  const options = { action };
  const valueFlags = { '--variant': 'variant', '--version': 'version', '--directory': 'directory', '--repo': 'repo', '--target': 'target' };
  const boolFlags = { '--dry-run': 'dryRun', '--confirm-draft': 'confirmDraft', '--acknowledge-limitations': 'acknowledgeLimitations' };
  for (let i = 0; i < rest.length; i++) {
    const key = valueFlags[rest[i]] ?? boolFlags[rest[i]];
    requireThat(key && !(key in options), 'unknown or duplicate option');
    if (rest[i] in valueFlags) {
      requireThat(rest[i + 1] && !rest[i + 1].startsWith('--'), 'option value missing');
      options[key] = rest[++i];
    } else options[key] = true;
  }
  validateVersion(options.version);
  validateVariant(options.variant);
  if (action !== 'build') {
    requireThat(options.directory, '--directory is required');
    options.directory = resolve(options.directory);
  } else requireThat(!options.directory, 'build output is always a fresh test-results run in this repository; no arbitrary overwrite');
  if (action === 'publish') {
    validateRepo(options.repo); validateTarget(options.target);
    requireThat(!(options.dryRun && options.confirmDraft), 'dry-run and confirm-draft are mutually exclusive');
    requireThat(options.dryRun || (options.confirmDraft && options.acknowledgeLimitations), 'choose --dry-run or both --confirm-draft --acknowledge-limitations');
  } else requireThat(!options.repo && !options.target && !options.dryRun && !options.confirmDraft && !options.acknowledgeLimitations, 'publish flags only apply to publish');
  return options;
}

async function main() {
  const options = parse(process.argv.slice(2));
  if (options.help) { console.log(usage); return; }
  const spec = selectSpec(await json(join(tool, 'spec.json')), options.variant);
  const python = join(tool, '.venv/bin/python');
  const outputRoot = join(project, 'test-results');
  await mkdir(outputRoot, { recursive: true });
  const run = join(outputRoot, `aec-model-${options.variant}-${new Date().toISOString().replaceAll(':', '-')}-${randomUUID()}`);
  await mkdir(run, { mode: 0o700 });
  const lock = join(outputRoot, '.runner-lock');
  const controller = new AbortController();
  const onSignal = () => controller.abort();
  let ownsLock = false;
  let summary = { action: options.action, variant: options.variant, version: options.version, status: 'RUNNING', fullAcceptancePassed: false,
    applicationSourceTouched: false, scope: 'standalone model artifact repository only' };
  const writeSummary = () => writeFile(join(run, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  try {
    await mkdir(lock, { mode: 0o700 });
    ownsLock = true;
    await writeFile(join(lock, 'owner.json'), JSON.stringify({ pid: process.pid, startedAt: new Date().toISOString(), runId: run, tool: 'aec-model' }), { mode: 0o600 });
    process.on('SIGINT', onSignal); process.on('SIGTERM', onSignal);
    const before = await toolHashes(tool);
    const verification = async (directory, version) => {
      await access(python);
      const result = await runProcess(python, [join(tool, 'builder.py'), 'verify', '--variant', options.variant, '--version', version, '--directory', directory],
        { cwd: project, env: safeEnv(), signal: controller.signal, timeoutMs: 180_000 });
      // This child has no credentials and processes only allowlisted public model assets.
      await writeFile(join(run, 'verification.log'), result.stdout + result.stderr);
      requireThat(result.code === 0, 'ONNX verification failed; inspect local verification.log');
    };
    if (options.action === 'build') {
      requireThat(process.platform === 'darwin' && process.arch === 'arm64', 'v1 builder requires macOS arm64');
      const version = await runProcess('uv', ['--version'], { env: safeEnv(), signal: controller.signal, timeoutMs: 10_000 });
      requireThat(version.code === 0 && version.stdout.match(/^uv (\S+)/)?.[1] === spec.recipe.uv, `requires uv ${spec.recipe.uv}`);
      const sync = await runProcess('uv', ['sync', '--locked', '--no-dev', '--no-build', '--no-config', '--default-index', 'https://pypi.org/simple', '--project', tool, '--python', spec.recipe.python],
        { cwd: project, env: safeEnv(), signal: controller.signal, timeoutMs: 900_000 });
      await writeFile(join(run, 'environment-setup.log'), sync.stdout + sync.stderr);
      requireThat(sync.code === 0, 'locked environment setup failed; inspect local environment-setup.log');
      const built = await runProcess(python, [join(tool, 'builder.py'), 'build', '--variant', options.variant, '--version', options.version, '--directory', run],
        { cwd: project, env: safeEnv(), signal: controller.signal, timeoutMs: 600_000 });
      await writeFile(join(run, 'build.log'), built.stdout + built.stderr);
      requireThat(built.code === 0, 'artifact build failed; inspect local build.log (partial files are not a release)');
      const directory = join(run, 'bundle');
      const bundle = await inspectBundle(directory, options.version, spec, before);
      summary = { ...summary, status: 'PASS', directory, mandatoryGatesPassed: true, diagnostics: bundle.validation.crossRuntimeDiagnostics,
        published: false, next: 'Inspect diagnostics; use explicit publish --dry-run before any Draft upload.' };
    } else if (options.action === 'verify') {
      const bundle = await inspectBundle(options.directory, options.version, spec, before);
      await verification(options.directory, options.version);
      const after = await inspectBundle(options.directory, options.version, spec, before);
      requireThat(isDeepStrictEqual(bundle.files, after.files), 'bundle changed during verification');
      summary = { ...summary, status: 'PASS', mandatoryGatesPassed: true, diagnostics: after.validation.crossRuntimeDiagnostics, published: false };
    } else {
      const published = await publishDraft(options, { spec, expectedTools: before, verify: verification,
        client: githubClient({ signal: controller.signal }), work: run,
        onCreated: async info => { summary.createdDraft = info; await writeSummary(); } });
      summary = { ...summary, status: 'PASS', publication: published };
    }
    requireThat(isDeepStrictEqual(before, await toolHashes(tool)), 'tool source changed during execution');
  } catch (error) {
    summary = { ...summary, status: error.code === 'EEXIST' ? 'BLOCKED' : 'FAIL', error: error.message };
    process.exitCode = summary.status === 'BLOCKED' ? 2 : 1;
  } finally {
    process.removeListener('SIGINT', onSignal); process.removeListener('SIGTERM', onSignal);
    if (ownsLock) {
      try {
        const owner = await json(join(lock, 'owner.json'));
        requireThat(owner.pid === process.pid, 'workspace lock ownership changed');
        await rm(lock, { recursive: true });
        summary.cleanup = 'PASS';
      } catch { summary.cleanup = 'FAIL'; summary.status = 'FAIL'; process.exitCode = 1; }
    }
    await writeSummary();
    console.log(`AEC model ${options.action}: ${summary.status}\nReport: ${join(run, 'summary.json')}`);
    if (summary.directory) console.log(`Bundle: ${summary.directory}`);
    if (summary.error) console.error(summary.error);
  }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
