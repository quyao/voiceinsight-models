import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink, readdir, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, basename, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { assetsFor, selectSpec, validateVariant, GATES, sha, json, toolHashes, inspectBundle, publishDraft, validateVersion, validateRepo, validateTarget,
  safeEnv, parseApiResponse, githubClient, runProcess } from '../tools/aec-model/release.mjs';

const tool = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'tools/aec-model');
const baseSpec = await json(join(tool, 'spec.json'));
const spec = selectSpec(baseSpec, '256');
const ASSETS = assetsFor('256');
const tools = await toolHashes(tool);
const version = '1.0.0-test.1';
const target = 'a'.repeat(40);
const put = (path, data) => writeFile(path, JSON.stringify(data, null, 2) + '\n');

async function fixture(t, variant = '256') {
  const spec = selectSpec(baseSpec, variant);
  const ASSETS = assetsFor(variant);
  const root = await mkdtemp(join(tmpdir(), 'vi-aec-tool-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const directory = join(root, 'bundle');
  await mkdir(directory);
  const validation = { schemaVersion: 1, model: spec.id, variant, version, mandatoryGatesPassed: true,
    gates: Object.fromEntries(GATES.map(k => [k, true])), toolHashes: tools,
    models: [], crossRuntimeDiagnostics: [
      { status: 'FAIL', tolerance: spec.crossRuntimeTolerance, outputMaxError: 0.0002, stateMaxError: 0 },
      { status: 'PASS', tolerance: spec.crossRuntimeTolerance, outputMaxError: 0, stateMaxError: 0 },
    ], limitations: ['not official ONNX', 'not acoustic acceptance', 'cross-runtime failure retained'] };
  for (let i = 0; i < 2; i++) {
    const name = `model_${variant}_${i + 1}.onnx`;
    // Fake bytes ONLY for transport/manifest unit tests. Real CLI always calls Python ONNX verifier.
    const bytes = Buffer.from(`not-a-real-model-${i}`);
    await writeFile(join(directory, name), bytes);
    validation.models.push({ file: name, size: bytes.length, sha256: sha(bytes), graphFingerprint: spec.expectedGraphs[i],
      resetMaxAbsoluteError: 0, framesPerPass: 120, passes: 2 });
  }
  await writeFile(join(directory, 'LICENSE-DTLN-aec'), await readFile(join(tool, 'DTLN-LICENSE')));
  await writeFile(join(directory, 'NOTICE'), 'VoiceInsight-maintained conversion, not official ONNX\n');
  await put(join(directory, 'validation-summary.json'), validation);
  const manifest = { schemaVersion: 1, model: spec.id, variant, version, upstream: spec.upstream, recipe: spec.recipe,
    contract: spec.contract, toolHashes: tools, sources: spec.sources.map(e => ({ ...e,
      url: `https://raw.githubusercontent.com/breizhn/DTLN-aec/${spec.upstream.commit}/${e.path}` })), files: {} };
  const refresh = async () => {
    for (const name of ASSETS.filter(n => n !== 'manifest.json')) {
      const data = await readFile(join(directory, name));
      manifest.files[name] = { size: data.length, sha256: sha(data) };
    }
    await put(join(directory, 'manifest.json'), manifest);
  };
  await refresh();
  const options = { repo: 'example/models', variant, version, target, directory, confirmDraft: true, acknowledgeLimitations: true };
  const calls = [], uploads = [];
  const client = {
    async api(method, path, data) {
      calls.push({ method, path, data });
      if (path === 'repos/example/models') return { status: 200, body: { permissions: { push: true } } };
      if (path.includes('/commits/')) return { status: 200, body: { sha: target } };
      if (path.includes('/releases/tags/') || path.includes('/git/ref/tags/')) return { status: 404, body: {} };
      if (method === 'POST') return { status: 201, body: { id: 123, draft: true, tag_name: `aec-dtln-${variant}-v${version}` } };
      if (path.endsWith('/assets?per_page=100')) {
        const bundle = await inspectBundle(directory, version, spec, tools);
        return { status: 200, body: ASSETS.map(name => ({ name, size: bundle.files[name].size, state: 'uploaded', digest: `sha256:${bundle.files[name].sha256}` })) };
      }
      return { status: 200, body: { id: 123, draft: true, tag_name: `aec-dtln-${variant}-v${version}` } };
    },
    async upload(repo, id, path) { uploads.push({ repo, id, name: basename(path) }); },
  };
  let verifications = 0;
  const deps = { spec, expectedTools: tools, work: root, client, verify: async path => {
    assert.notEqual(path, directory); assert.deepEqual((await readdir(path)).sort(), ASSETS); verifications++;
  } };
  return { root, directory, spec, manifest, validation, refresh, options, calls, uploads, client, deps, verified: () => verifications };
}

test('rejects traversal, implicit targets, malformed versions and repositories', () => {
  for (const v of ['../1.0.0', 'latest', '01.0.0', '1.0.0;id', '-x', '1.0.0\n']) assert.throws(() => validateVersion(v));
  for (const r of ['https://github.com/o/r', '../x', 'o/r/extra', '--repo', 'o/r?token=x']) assert.throws(() => validateRepo(r));
  assert.throws(() => validateTarget('main'));
  assert.equal(validateVersion(version), version); assert.equal(validateTarget(target), target);
});

test('all three variants have disjoint assets/tags, explicit state shape and reject cross-variant bundles', async t => {
  const tags = new Set();
  for (const variant of ['128', '256', '512']) {
    const f = await fixture(t, variant);
    assert.deepEqual(f.spec.contract.stateShape, [1, 2, Number(variant), 2]);
    const result = await publishDraft({ ...f.options, dryRun: true }, f.deps);
    assert.equal(result.plan.variant, variant);
    assert(result.plan.assets.some(a => a.name === `model_${variant}_1.onnx`));
    assert.equal(result.plan.tag, `aec-dtln-${variant}-v${version}`);
    tags.add(result.plan.tag);
    const wrong = selectSpec(baseSpec, variant === '128' ? '512' : '128');
    await assert.rejects(inspectBundle(f.directory, version, wrong, tools));
  }
  assert.equal(tags.size, 3);
  for (const invalid of [undefined, '64', '1024', 256, '../128', '128\n']) assert.throws(() => validateVariant(invalid));
});

test('bundle whitelist and SHA checks reject extra files and tampering', async t => {
  const f = await fixture(t);
  await inspectBundle(f.directory, version, spec, tools);
  await writeFile(join(f.directory, 'token.txt'), 'not-a-real-secret');
  await assert.rejects(inspectBundle(f.directory, version, spec, tools), /six public/);
  await rm(join(f.directory, 'token.txt'));
  await writeFile(join(f.directory, 'model_256_1.onnx'), 'corrupt');
  await assert.rejects(inspectBundle(f.directory, version, spec, tools), /checksum/);
});

test('rejects symlink assets and bundle directory', async t => {
  const f = await fixture(t);
  await symlink(f.directory, join(f.root, 'alias'));
  await assert.rejects(inspectBundle(join(f.root, 'alias'), version, spec, tools), /real directory/);
  const path = join(f.directory, 'NOTICE');
  await rm(path); await symlink(join(tool, 'DTLN-LICENSE'), path);
  await assert.rejects(inspectBundle(f.directory, version, spec, tools), /invalid asset/);
});

test('rejects false validation, wrong recipe, stale tool source and model/report mismatch', async t => {
  for (const mutation of [
    f => { f.validation.mandatoryGatesPassed = false; },
    f => { f.validation.crossRuntimeDiagnostics[0].status = 'PASS'; },
    f => { f.validation.crossRuntimeDiagnostics[0].tolerance = 1; },
    f => { f.validation.models[0].sha256 = 'f'.repeat(64); },
    f => { f.manifest.upstream = { ...spec.upstream, commit: 'main' }; },
    f => { f.manifest.toolHashes = { ...tools, 'builder.py': 'f'.repeat(64) }; },
  ]) {
    const f = await fixture(t); mutation(f);
    await put(join(f.directory, 'validation-summary.json'), f.validation); await f.refresh();
    await assert.rejects(inspectBundle(f.directory, version, spec, tools));
  }
});

test('dry-run verifies frozen assets, preserves FAIL diagnostics, never calls GitHub', async t => {
  const f = await fixture(t);
  const result = await publishDraft({ ...f.options, dryRun: true, confirmDraft: false, acknowledgeLimitations: false }, f.deps);
  assert.equal(result.status, 'DRY_RUN'); assert.equal(result.networkCalls, 0);
  assert.equal(result.plan.diagnostics[0].status, 'FAIL');
  assert.equal(f.verified(), 1); assert.deepEqual(f.calls, []); assert.deepEqual(f.uploads, []);
  await assert.rejects(access(join(f.root, 'upload-snapshot')));
});

test('live publish requires explicit Draft and limitations confirmation', async t => {
  for (const missing of ['confirmDraft', 'acknowledgeLimitations']) {
    const f = await fixture(t);
    await assert.rejects(publishDraft({ ...f.options, [missing]: false }, f.deps), /requires/);
    assert.deepEqual(f.calls, []);
  }
});

test('real ONNX verifier is mandatory even when JSON gates say PASS', async t => {
  const f = await fixture(t);
  f.deps.verify = async () => { throw new Error('ONNX graph invalid'); };
  await assert.rejects(publishDraft(f.options, f.deps), /ONNX graph invalid/);
  assert.deepEqual(f.calls, []); assert.deepEqual(f.uploads, []);
});

test('snapshot modification during verification aborts before GitHub even in dry-run', async t => {
  const f = await fixture(t);
  f.deps.verify = async path => { await writeFile(join(path, 'NOTICE'), 'changed'); };
  await assert.rejects(publishDraft({ ...f.options, dryRun: true }, f.deps), /checksum|snapshot changed/);
  assert.deepEqual(f.calls, []);
});

test('existing release/tag, unknown HTTP failures and permission denial never mutate GitHub', async t => {
  for (const scenario of ['release', 'tag', 'auth', 'permission', 'target']) {
    const f = await fixture(t), original = f.client.api;
    f.client.api = async (...args) => {
      const result = await original(...args), path = args[1];
      if (scenario === 'permission' && path === 'repos/example/models') return { status: 200, body: { permissions: { push: false } } };
      if (scenario === 'target' && path.includes('/commits/')) return { status: 404 };
      if (path.includes('/releases/tags/') && ['release', 'auth'].includes(scenario)) return { status: scenario === 'release' ? 200 : 401 };
      if (scenario === 'tag' && path.includes('/git/ref/tags/')) return { status: 200 };
      return result;
    };
    await assert.rejects(publishDraft(f.options, f.deps));
    assert.equal(f.calls.some(c => c.method !== 'GET'), false); assert.deepEqual(f.uploads, []);
  }
});

test('successful transport uploads only six verified assets to exact Draft ID, never publishes', async t => {
  const f = await fixture(t), created = [];
  f.deps.onCreated = async x => created.push(x);
  const result = await publishDraft(f.options, f.deps);
  assert.equal(result.status, 'DRAFT_UPLOADED'); assert.equal(result.publiclyPublished, false);
  assert.equal(result.remoteHashesVerified, true); assert.equal(created[0].id, 123);
  const mutations = f.calls.filter(c => c.method !== 'GET');
  assert.equal(mutations.length, 1); assert.equal(mutations[0].method, 'POST');
  assert.equal(mutations[0].data.draft, true); assert.equal(mutations[0].data.make_latest, 'false');
  assert.equal(mutations[0].data.target_commitish, target);
  assert.deepEqual(f.uploads.map(x => x.name), ASSETS); assert(f.uploads.every(x => x.id === 123));
});

test('upload failures retain Draft, do not delete/retry and remove local snapshot', async t => {
  const f = await fixture(t);
  f.client.upload = async () => { throw new Error('secret-provider-diagnostic'); };
  await assert.rejects(publishDraft(f.options, f.deps), error => /ID 123 retained/.test(error.message) && !error.message.includes('secret-provider'));
  assert(!f.calls.some(c => ['DELETE', 'PATCH'].includes(c.method)));
  await assert.rejects(access(join(f.root, 'upload-snapshot')));
});

test('remote digest failure or external Draft state change cannot report success', async t => {
  for (const scenario of ['digest', 'published']) {
    const f = await fixture(t), original = f.client.api;
    f.client.api = async (...args) => {
      const result = await original(...args);
      if (scenario === 'digest' && Array.isArray(result.body)) delete result.body[0].digest;
      if (scenario === 'published' && args[1].endsWith('/releases/123')) result.body.draft = false;
      return result;
    };
    await assert.rejects(publishDraft(f.options, f.deps), /retained after failure/);
    if (scenario === 'published') assert.equal(f.uploads.length, 0);
  }
});

test('GitHub status parser fails closed; credentials only reach GitHub child environment', () => {
  assert.equal(parseApiResponse({ code: 1, stdout: 'HTTP/2.0 404 Not Found\r\nContent-Type: application/json\r\n\r\n{"message":"Not Found"}' }).status, 404);
  assert.throws(() => parseApiResponse({ code: 1, stdout: 'auth error token=private' }), /lacks an HTTP status/);
  assert.throws(() => parseApiResponse({ code: 0, stdout: 'HTTP/2.0 200 OK\n\nnot json' }), /invalid GitHub JSON/);
  const source = { PATH: '/bin', HOME: '/home/test', GH_TOKEN: 'fake-auth', OPENAI_API_KEY: 'fake-provider', GH_DEBUG: 'api', GH_HOST: 'evil.example', UV_INDEX: 'private' };
  assert.equal(safeEnv(source).GH_TOKEN, undefined);
  const gh = safeEnv(source, true);
  assert.equal(gh.GH_TOKEN, 'fake-auth'); assert.equal(gh.GH_HOST, 'github.com');
  for (const key of ['OPENAI_API_KEY', 'GH_DEBUG', 'UV_INDEX']) assert.equal(gh[key], undefined);
});

test('gh adapter uses stdin for release JSON and uploads by ID without clobber or credentials in argv', async () => {
  const calls = [];
  const client = githubClient({ run: async (...args) => { calls.push(args); return { code: 0, stdout: 'HTTP/2.0 201 Created\n\n{"id":123}' }; } });
  await client.api('POST', 'repos/example/models/releases', { draft: true });
  await client.upload('example/models', 123, '/tmp/model_256_1.onnx');
  assert.equal(calls[0][2].input, '{"draft":true}');
  assert(calls[1][1].includes('https://uploads.github.com/repos/example/models/releases/123/assets?name=model_256_1.onnx'));
  assert(!calls.flatMap(c => c[1]).includes('--clobber'));
});

test('successful parent cannot leave an owned detached-stdio descendant running', async t => {
  const code = `const {spawn}=require('node:child_process');const c=spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{stdio:'ignore'});console.log(c.pid);c.unref();`;
  const result = await runProcess(process.execPath, ['-e', code], { timeoutMs: 2000 });
  assert.equal(result.code, 0);
  const pid = Number(result.stdout.trim());
  assert(Number.isSafeInteger(pid) && pid > 0);
  t.after(() => { try { process.kill(pid, 'SIGKILL'); } catch {} });
  let alive = true;
  for (let i = 0; i < 40 && alive; i++) {
    try { process.kill(pid, 0); await new Promise(r => setTimeout(r, 25)); } catch (e) { if (e.code === 'ESRCH') alive = false; else throw e; }
  }
  assert.equal(alive, false);
});

test('bounded subprocess reports timeout/output overflow without leaking child stderr', async () => {
  await assert.rejects(runProcess(process.execPath, ['-e', 'process.stderr.write("fake-token");setInterval(()=>{},1000)'], { timeoutMs: 50 }), e => /timed out/.test(e.message) && !e.message.includes('fake-token'));
  await assert.rejects(runProcess(process.execPath, ['-e', 'process.stdout.write("x".repeat(50000));setInterval(()=>{},1000)'], { maxBytes: 100, timeoutMs: 1000 }), /output limit/);
});
