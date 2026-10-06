import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, readdir, lstat, mkdir, copyFile, rm, chmod } from 'node:fs/promises';
import { join, basename } from 'node:path';
import { isDeepStrictEqual } from 'node:util';

export function validateVariant(value) {
  requireThat(['128', '256', '512'].includes(value), '--variant must explicitly be 128, 256 or 512');
  return value;
}
export function assetsFor(variant) {
  validateVariant(variant);
  return ['LICENSE-DTLN-aec', 'NOTICE', 'manifest.json', `model_${variant}_1.onnx`, `model_${variant}_2.onnx`, 'validation-summary.json'].sort();
}
export function selectSpec(base, variant) {
  validateVariant(variant);
  const { variants, licenseSource, ...common } = base;
  requireThat(variants[variant], 'variant recipe is missing');
  return { ...common, id: `dtln-aec-${variant}`, variant,
    contract: { ...common.contract, stateShape: [1, 2, Number(variant), 2] },
    sources: [...variants[variant].sources, licenseSource], expectedGraphs: variants[variant].expectedGraphs };
}
export const TOOL_FILES = ['spec.json', 'builder.py', 'pyproject.toml', 'uv.lock'];
export const GATES = ['sourceHashes', 'authorGraphAndWeights', 'interfaceContract', 'finiteNonSilentInference', 'recurrentReset'];
export function requireThat(value, message) { if (!value) throw new Error(message); }
export function validateVersion(value) {
  requireThat(typeof value === 'string' && value === value.trim() && value.length <= 64 && /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[a-z0-9]+(?:[.-][a-z0-9]+)*)?$/.test(value), 'invalid version');
  return value;
}
export function validateRepo(value) {
  requireThat(typeof value === 'string' && value === value.trim() && /^[A-Za-z0-9][A-Za-z0-9-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(value) && value.length <= 200, 'expected GitHub owner/repository');
  return value;
}
export function validateTarget(value) {
  requireThat(typeof value === 'string' && value.length === 40 && /^[a-f0-9]{40}$/.test(value), '--target must be an explicit full Git commit SHA');
  return value;
}
export const sha = bytes => createHash('sha256').update(bytes).digest('hex');
export const json = path => readFile(path, 'utf8').then(JSON.parse);
export async function toolHashes(root) {
  return Object.fromEntries(await Promise.all(TOOL_FILES.map(async name => [name, sha(await readFile(join(root, name)))])));
}
export function safeEnv(source = process.env, github = false) {
  const env = {};
  for (const key of ['PATH', 'HOME', 'TMPDIR', 'SystemRoot']) if (source[key]) env[key] = source[key];
  Object.assign(env, { TF_CPP_MIN_LOG_LEVEL: '2', OMP_NUM_THREADS: '1', OPENBLAS_NUM_THREADS: '1', MKL_NUM_THREADS: '1', CUDA_VISIBLE_DEVICES: '', PYTHONNOUSERSITE: '1' });
  if (github) {
    for (const key of ['GH_TOKEN', 'GITHUB_TOKEN', 'GH_CONFIG_DIR']) if (source[key]) env[key] = source[key];
    Object.assign(env, { GH_HOST: 'github.com', GH_PROMPT_DISABLED: '1', GIT_TERMINAL_PROMPT: '0', GH_PAGER: 'cat', NO_COLOR: '1' });
  }
  return env;
}

// Bounded process groups: timeout/interruption must not leave a converter/upload running.
// Never surface child stderr (gh/uv can include authentication material).
export function runProcess(command, args, { cwd, env = safeEnv(), input, timeoutMs = 120_000, signal, maxBytes = 8 * 1024 * 1024 } = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { cwd, env, detached: process.platform !== 'win32', stdio: ['pipe', 'pipe', 'pipe'] });
    let bytes = 0, out = [], err = [], stopped = null, timer, escalation;
    const kill = sig => {
      try { if (process.platform !== 'win32') process.kill(-child.pid, sig); else child.kill(sig); } catch (e) { if (e.code !== 'ESRCH') child.kill(sig); }
    };
    const stop = reason => {
      if (stopped) return;
      stopped = reason; kill('SIGTERM');
      escalation = setTimeout(() => kill('SIGKILL'), 1500);
    };
    const abort = () => stop('interrupted');
    const collect = target => chunk => { bytes += chunk.length; if (bytes > maxBytes) stop('output limit exceeded'); else target.push(chunk); };
    child.stdout.on('data', collect(out));
    child.stderr.on('data', collect(err));
    child.stdin.on('error', () => {});
    child.stdin.end(input ?? '');
    child.on('error', () => { clearTimeout(timer); clearTimeout(escalation); signal?.removeEventListener('abort', abort); reject(new Error(`cannot start ${command}`)); });
    child.on('close', (code, childSignal) => {
      clearTimeout(timer); clearTimeout(escalation); signal?.removeEventListener('abort', abort);
      // Descendants may outlive a successful parent with inherited stdio closed.
      // This group belongs exclusively to this invocation, never to other tasks.
      kill('SIGKILL');
      if (stopped) {
        reject(new Error(`${command}: ${stopped}`));
      } else resolvePromise({ code, signal: childSignal, stdout: Buffer.concat(out).toString('utf8'), stderr: Buffer.concat(err).toString('utf8') });
    });
    timer = setTimeout(() => stop('timed out'), timeoutMs);
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
  });
}

export async function inspectBundle(directory, version, spec, expectedTools) {
  validateVersion(version);
  const assets = assetsFor(spec.variant);
  const root = await lstat(directory);
  requireThat(root.isDirectory() && !root.isSymbolicLink(), 'bundle must be a real directory');
  const names = (await readdir(directory)).sort();
  requireThat(isDeepStrictEqual(names, assets), 'bundle must contain exactly the six public release assets');
  const files = {};
  for (const name of names) {
    const path = join(directory, name), stat = await lstat(path);
    requireThat(stat.isFile() && !stat.isSymbolicLink() && stat.size > 0 && stat.size <= 32 * 1024 * 1024, `invalid asset: ${name}`);
    const data = await readFile(path);
    files[name] = { size: data.length, sha256: sha(data) };
  }
  const manifest = await json(join(directory, 'manifest.json'));
  const validation = await json(join(directory, 'validation-summary.json'));
  requireThat(manifest.schemaVersion === 1 && manifest.model === spec.id && manifest.variant === spec.variant && manifest.version === version, 'manifest identity mismatch');
  for (const key of ['upstream', 'recipe', 'contract']) requireThat(isDeepStrictEqual(manifest[key], spec[key]), `manifest ${key} mismatch`);
  const sources = spec.sources.map(entry => ({ ...entry, url: `https://raw.githubusercontent.com/breizhn/DTLN-aec/${spec.upstream.commit}/${entry.path}` }));
  requireThat(isDeepStrictEqual(manifest.sources, sources), 'unexpected upstream source');
  requireThat(isDeepStrictEqual(manifest.toolHashes, expectedTools) && isDeepStrictEqual(validation.toolHashes, expectedTools), 'tool recipe changed; use the exact builder revision');
  requireThat(isDeepStrictEqual(Object.keys(manifest.files).sort(), assets.filter(n => n !== 'manifest.json')), 'manifest asset allowlist mismatch');
  for (const [name, expected] of Object.entries(manifest.files)) requireThat(isDeepStrictEqual(expected, files[name]), `checksum mismatch: ${name}`);
  requireThat(isDeepStrictEqual(files['LICENSE-DTLN-aec'], { size: spec.sources[2].size, sha256: spec.sources[2].sha256 }), 'license mismatch');
  requireThat(validation.schemaVersion === 1 && validation.model === spec.id && validation.variant === spec.variant && validation.version === version, 'validation identity mismatch');
  requireThat(validation.mandatoryGatesPassed === true && isDeepStrictEqual(Object.keys(validation.gates).sort(), [...GATES].sort()) && Object.values(validation.gates).every(v => v === true), 'mandatory validation did not pass');
  requireThat(validation.models?.length === 2 && validation.crossRuntimeDiagnostics?.length === 2 && validation.limitations?.length >= 3, 'incomplete validation report');
  for (let i = 0; i < 2; i++) {
    const model = validation.models[i];
    const name = `model_${spec.variant}_${i + 1}.onnx`;
    requireThat(model.file === name && model.sha256 === files[name].sha256 && model.size === files[name].size &&
      isDeepStrictEqual(model.graphFingerprint, spec.expectedGraphs[i]) && model.resetMaxAbsoluteError >= 0 && model.resetMaxAbsoluteError <= 1e-6 &&
      model.framesPerPass === 120 && model.passes === 2, 'model validation is not bound to these artifacts');
    const d = validation.crossRuntimeDiagnostics[i];
    requireThat(d.tolerance === spec.crossRuntimeTolerance && [d.outputMaxError, d.stateMaxError].every(x => Number.isFinite(x) && x >= 0), 'invalid diagnostic metrics');
    requireThat(d.status === (Math.max(d.outputMaxError, d.stateMaxError) <= spec.crossRuntimeTolerance ? 'PASS' : 'FAIL'), 'false diagnostic verdict');
  }
  return { manifest, validation, files };
}

export function makePlan({ repo, version, target }, bundle) {
  validateRepo(repo); validateVersion(version); validateTarget(target);
  const variant = validateVariant(bundle.manifest.variant);
  return { repository: repo, variant, tag: `aec-dtln-${variant}-v${version}`, targetCommit: target, draft: true, makeLatest: false,
    assets: assetsFor(variant).map(name => ({ name, ...bundle.files[name] })),
    diagnostics: bundle.validation.crossRuntimeDiagnostics,
    limitations: bundle.validation.limitations,
    notice: 'VoiceInsight-maintained ONNX conversion of author weights, NOT official author-published ONNX. No full acoustic/runtime-switch acceptance.' };
}

export function parseApiResponse(result) {
  // gh api --include returns HTTP headers even for non-2xx; fail closed if missing.
  const normalized = result.stdout.replaceAll('\r\n', '\n');
  const boundary = normalized.indexOf('\n\n');
  const match = normalized.split('\n')[0].match(/^HTTP\/[^\s]+ (\d{3})[^\n]*$/);
  requireThat(match && boundary >= 0, 'GitHub response lacks an HTTP status; authentication/network failure');
  const status = Number(match[1]);
  const payload = normalized.slice(boundary + 2);
  let body;
  try { body = payload.trim() ? JSON.parse(payload) : null; } catch { throw new Error('invalid GitHub JSON response'); }
  requireThat(result.code === 0 || status >= 400, 'GitHub command failed');
  return { status, body };
}
export function githubClient({ run = runProcess, signal } = {}) {
  return {
    async api(method, path, data) {
      const args = ['api', '--hostname', 'github.com', '--include', '--method', method, path];
      if (data !== undefined) args.push('--input', '-');
      const result = await run('gh', args, { env: safeEnv(process.env, true), input: data === undefined ? undefined : JSON.stringify(data), timeoutMs: 60_000, signal });
      return parseApiResponse(result);
    },
    async upload(repo, releaseId, path) {
      // Address the exact created ID, not a tag that another process could rebind.
      const url = `https://uploads.github.com/repos/${repo}/releases/${releaseId}/assets?name=${encodeURIComponent(basename(path))}`;
      const result = await run('gh', ['api', '--hostname', 'github.com', '--method', 'POST', url,
        '--header', 'Content-Type: application/octet-stream', '--input', path], { env: safeEnv(process.env, true), timeoutMs: 120_000, signal });
      requireThat(result.code === 0, 'GitHub upload failed; partial Draft retained; no overwrite/retry');
    },
  };
}

export async function publishDraft(options, { spec, expectedTools, verify, client, work, onCreated = async () => {} }) {
  // Snapshot a fixed allowlist. Verification and upload always use this snapshot,
  // not mutable user paths. Do not accidentally upload logs/source caches/secrets.
  const assets = assetsFor(spec.variant);
  const initial = await inspectBundle(options.directory, options.version, spec, expectedTools);
  const plan = makePlan(options, initial);
  if (!options.dryRun) {
    requireThat(options.confirmDraft, 'publishing requires --confirm-draft');
    requireThat(options.acknowledgeLimitations, 'publishing requires --acknowledge-limitations (quality/cross-runtime boundaries)');
  }
  const snapshot = join(work, 'upload-snapshot');
  await mkdir(snapshot, { mode: 0o700 });
  try {
    for (const name of assets) await copyFile(join(options.directory, name), join(snapshot, name));
    const frozen = await inspectBundle(snapshot, options.version, spec, expectedTools);
    requireThat(isDeepStrictEqual(initial.files, frozen.files), 'bundle changed while snapshotting');
    await verify(snapshot, options.version); // Actual ONNX graph + inference, never just a JSON passed flag.
    requireThat(isDeepStrictEqual((await inspectBundle(snapshot, options.version, spec, expectedTools)).files, frozen.files), 'snapshot changed during verification');
    for (const name of assets) await chmod(join(snapshot, name), 0o400);
    if (options.dryRun) return { status: 'DRY_RUN', networkCalls: 0, plan };
    // Do not call gh at all in dry-run. No credentials leave the machine before this point.
    const repo = await client.api('GET', `repos/${options.repo}`);
    requireThat(repo.status === 200 && repo.body.permissions?.push === true, 'GitHub repository unavailable or lacks push permission');
    const commit = await client.api('GET', `repos/${options.repo}/commits/${options.target}`);
    requireThat(commit.status === 200 && commit.body.sha === options.target, 'target commit is not available in this repository');
    const release = await client.api('GET', `repos/${options.repo}/releases/tags/${plan.tag}`);
    requireThat(release.status === 404, release.status === 200 ? 'release already exists; refusing overwrite/resume' : 'cannot verify release absence');
    const tag = await client.api('GET', `repos/${options.repo}/git/ref/tags/${plan.tag}`);
    requireThat(tag.status === 404, tag.status === 200 ? 'tag already exists; refusing reuse' : 'cannot verify tag absence');
    const created = await client.api('POST', `repos/${options.repo}/releases`, {
      tag_name: plan.tag, target_commitish: options.target, name: `DTLN-aec ${spec.variant} ONNX ${options.version}`,
      body: `${plan.notice}\n\nMandatory ONNX packaging gates passed. Inspect validation-summary.json for cross-runtime diagnostics (including FAIL) and limitations.\n\n${plan.limitations.join('\n')}`,
      draft: true, prerelease: options.version.includes('-'), make_latest: 'false',
    });
    requireThat(created.status === 201 && Number.isSafeInteger(created.body.id) && created.body.draft === true && created.body.tag_name === plan.tag, 'unexpected Draft creation response; inspect GitHub before retrying');
    const id = created.body.id;
    try {
      await onCreated({ id, repository: options.repo, tag: plan.tag, draft: true });
      for (const name of assets) {
        requireThat(isDeepStrictEqual((await inspectBundle(snapshot, options.version, spec, expectedTools)).files, frozen.files), 'snapshot changed before upload');
        // Use the release ID for state checks; never upload to a published release.
        const state = await client.api('GET', `repos/${options.repo}/releases/${id}`);
        requireThat(state.status === 200 && state.body.id === id && state.body.draft === true && state.body.tag_name === plan.tag, 'Draft changed externally; stop uploading');
        await client.upload(options.repo, id, join(snapshot, name));
      }
      const state = await client.api('GET', `repos/${options.repo}/releases/${id}`);
      requireThat(state.status === 200 && state.body.draft === true && state.body.id === id && state.body.tag_name === plan.tag, 'Draft changed externally after upload');
      const remoteAssets = await client.api('GET', `repos/${options.repo}/releases/${id}/assets?per_page=100`);
      requireThat(remoteAssets.status === 200 && Array.isArray(remoteAssets.body) && remoteAssets.body.length === assets.length, 'remote asset list mismatch');
      requireThat(isDeepStrictEqual(remoteAssets.body.map(a => a.name).sort(), assets), 'remote asset names mismatch');
      for (const asset of remoteAssets.body) {
        const expected = frozen.files[asset.name];
        requireThat(asset.state === 'uploaded' && asset.size === expected.size && asset.digest === `sha256:${expected.sha256}`, `remote checksum missing/mismatch: ${asset.name}`);
      }
      return { status: 'DRAFT_UPLOADED', releaseId: id, plan, remoteHashesVerified: true, publiclyPublished: false };
    } catch {
      // Never delete somebody else's release/tag or overwrite assets on retries.
      throw new Error(`Draft ${options.repo} release ID ${id} retained after failure; inspect manually. No automatic delete, retry or public publish.`);
    }
  } finally {
    await rm(snapshot, { recursive: true, force: true });
  }
}
