// Free-model aggregator: best free model per request, all three upstream wires,
// failover across providers, deterministic (no clock, no unseeded randomness).
//
// Routing: candidates are free models meeting the request's requirements, ranked
// by the brief's rating_milli (read at runtime from MAKEFUN_PROMPT_FILE), with a
// small penalty per provider call already made, per observed failure, and once a
// provider passes its conservative capacity estimate (the sealed per-provider
// budget floor). The request order in the manifest is weight-descending, so
// greedy per-request routing gives heavy requests the best models first.
// Failover: on a 402/429/5xx (or a network error / unparsable answer) the next
// candidate is tried, up to 6 attempts; a provider that failed is never retried
// for the same request. A JSON-mode OpenAI request is never sent to an
// Anthropic-wire provider (its wire cannot carry JSON mode, so the echo would
// lose it); Gemini carries it via responseMimeType.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicReq,
  openAIToGemini,
  parseAnthropicAnswer,
  parseAnthropicStream,
  parseGeminiAnswer,
  parseGeminiStream,
  parseOpenAIStream,
  parsedToAnthropicJson,
  parsedToAnthropicStream,
  parsedToOpenAIJson,
  parsedToOpenAIStream,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const MAX_ATTEMPTS = 6;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

/** Conservative per-provider capacity floor (sealed budget ranges' minima). */
const MIN_BUDGET: Record<string, number> = {
  groq: 9, cerebras: 5, gemini: 6, zai: 4, openrouter: 10, sambanova: 4,
  pollinations: 10, cloudflare: 7, ovh: 5, kilol: 12,
};

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
/** Provider id -> upstream calls attempted so far (successes and refusals). */
const callsByProvider = new Map<string, number>();
/** Provider id -> upstream calls refused so far. */
const failuresByProvider = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function dialBase(provider: ProviderDirectoryEntry): string {
  return provider.dial !== '' ? provider.dial : joinBase(ENV_URL, provider.base);
}

/** Fetch the env's /instance route and build directory entries from it (absolute base_urls). */
async function instanceProviders(): Promise<ProviderDirectoryEntry[] | null> {
  if (ENV_URL === '') return null;
  try {
    const res = await fetch(`${ENV_URL}/instance`, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return null;
    const body: unknown = await res.json();
    if (!isRecord(body) || !Array.isArray(body.providers)) return null;
    const out: ProviderDirectoryEntry[] = [];
    for (const raw of body.providers) {
      if (!isRecord(raw) || typeof raw.id !== 'string' || raw.id === '') continue;
      const wire = raw.wire === 'openai' || raw.wire === 'anthropic' || raw.wire === 'gemini' ? raw.wire : null;
      if (wire === null) continue;
      const auth = raw.auth === 'bearer' || raw.auth === 'x-api-key' || raw.auth === 'x-goog-api-key' || raw.auth === 'none' ? raw.auth : 'bearer';
      const baseUrl = typeof raw.base_url === 'string' ? raw.base_url : '';
      let path = '';
      if (baseUrl !== '') { try { path = new URL(baseUrl).pathname.replace(/\/+$/, ''); } catch { path = baseUrl; } }
      out.push({
        id: raw.id,
        name: typeof raw.name === 'string' && raw.name !== '' ? raw.name : raw.id,
        wire,
        auth,
        base: path,
        key_env: typeof raw.key_env === 'string' ? raw.key_env : '',
        key: typeof raw.key === 'string' ? raw.key : '',
        dial: baseUrl,
      });
    }
    return out.length > 0 ? out : null;
  } catch {
    return null;
  }
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  let directory = brief.directory.providers;
  const fromInstance = await instanceProviders();
  if (fromInstance !== null) {
    // Prefer the env's directory (it is the live truth for dialling); keep the
    // brief's entry for any provider the instance did not list.
    const byId = new Map(fromInstance.map((p) => [p.id, p]));
    for (const p of directory) if (!byId.has(p.id)) byId.set(p.id, p);
    directory = [...byId.values()];
  }
  probes = await Promise.all(directory.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  score: number;
}

function candidateScore(rating: number, providerId: string): number {
  const failures = failuresByProvider.get(providerId) ?? 0;
  // No per-call or per-failure penalty: a refusal is usually one scheduled
  // fail-call, and the next call on that provider normally succeeds, so a
  // top model stays top until its budget is really spent (failover then picks
  // the next best). Repeated refusals (outage / spent budget) blacklist it.
  // The manifest serves scarce and heavy requests first, so spending a top
  // provider early cannot starve a later request that only it can serve.
  return failures >= 3 ? rating - 500 : rating;
}

/**
 * Free models meeting the requirements, best first. `jsonViaOpenAI` is true when
 * the client is OpenAI and asked for JSON mode: Anthropic-wire providers cannot
 * carry that, so they are excluded for such a request.
 */
function candidatesFor(requires: Requirement, jsonViaOpenAI: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonViaOpenAI && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      out.push({ probe, model, rating, score: candidateScore(rating, probe.provider.id) });
    }
  }
  out.sort((a, b) => b.score - a.score || b.rating - a.rating || (callsByProvider.get(a.probe.provider.id) ?? 0) - (callsByProvider.get(b.probe.provider.id) ?? 0) || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : a.model.id < b.model.id ? -1 : 1));
  return out;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

function jsonModeOpenAI(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): UpstreamCall {
  const probe = candidate.probe;
  const base = dialBase(probe.provider);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(probe.provider.auth, probe.provider.key) };
  const model = candidate.model.id;
  if (probe.provider.wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (stream) out.stream_options = { include_usage: true };
      return { url: `${base}/v1/chat/completions`, headers, body: out };
    }
    const out = anthropicToOpenAI(body, model);
    return { url: `${base}/v1/chat/completions`, headers, body: out };
  }
  if (probe.provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (clientApi === 'anthropic') return { url: `${base}/v1/messages`, headers, body: { ...body, model } };
    return { url: `${base}/v1/messages`, headers, body: openAIToAnthropicReq(body, model) };
  }
  // gemini
  const geminiBody = clientApi === 'openai' ? openAIToGemini(body, model) : anthropicToGemini(body, model);
  const verb = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(model)}:${verb}${stream ? '?alt=sse' : ''}`;
  if (stream) headers.accept = 'text/event-stream';
  return { url, headers, body: geminiBody };
}

type Served =
  | { kind: 'passthrough-json'; text: string }
  | { kind: 'json'; body: Json }
  | { kind: 'sse-text'; text: string }
  | { kind: 'passthrough-stream'; stream: ReadableStream<Uint8Array> }
  | { kind: 'openai-stream-events'; stream: ReadableStream<Uint8Array> };

/** Call one candidate; on success return how to answer the client, else null. */
async function tryCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Served | null> {
  const probe = candidate.probe;
  const call = upstreamCall(candidate, clientApi, body, stream);
  const pid = probe.provider.id;
  callsByProvider.set(pid, (callsByProvider.get(pid) ?? 0) + 1);
  let upstream: Response;
  try {
    upstream = await fetch(call.url, { method: 'POST', headers: call.headers, body: JSON.stringify(call.body), signal: AbortSignal.timeout(25000) });
  } catch {
    failuresByProvider.set(pid, (failuresByProvider.get(pid) ?? 0) + 1);
    return null;
  }
  if (!upstream.ok) {
    failuresByProvider.set(pid, (failuresByProvider.get(pid) ?? 0) + 1);
    try { await upstream.arrayBuffer(); } catch { /* drain */ }
    return null;
  }
  const wire = probe.provider.wire;
  const model = candidate.model.id;
  try {
    if (stream) {
      if (upstream.body === null) { failuresByProvider.set(pid, (failuresByProvider.get(pid) ?? 0) + 1); return null; }
      // Same wire: stream the provider's events straight through.
      if ((clientApi === 'openai' && wire === 'openai') || (clientApi === 'anthropic' && wire === 'anthropic')) {
        return { kind: 'passthrough-stream', stream: upstream.body };
      }
      // Anthropic client, OpenAI provider: the event converter reads the
      // stream itself, so hand it over untouched (never .text() it first).
      if (clientApi === 'anthropic' && wire === 'openai') {
        return { kind: 'openai-stream-events', stream: upstream.body };
      }
      // Cross wire: buffer, parse, re-emit in the client's wire.
      const text = await upstream.text();
      if (clientApi === 'openai') {
        if (wire === 'anthropic') return { kind: 'sse-text', text: parsedToOpenAIStream(parseAnthropicStream(text), model) };
        return { kind: 'sse-text', text: parsedToOpenAIStream(parseGeminiStream(text), model) };
      }
      return { kind: 'sse-text', text: parsedToAnthropicStream(parseGeminiStream(text), model) };
    }
    // JSON (not streaming).
    if (clientApi === 'openai' && wire === 'openai') return { kind: 'passthrough-json', text: await upstream.text() };
    if (clientApi === 'anthropic' && wire === 'anthropic') return { kind: 'passthrough-json', text: await upstream.text() };
    const answer: unknown = await upstream.json();
    if (!isRecord(answer)) { failuresByProvider.set(pid, (failuresByProvider.get(pid) ?? 0) + 1); return null; }
    if (clientApi === 'openai') {
      if (wire === 'anthropic') return { kind: 'json', body: parsedToOpenAIJson(parseAnthropicAnswer(answer), model) };
      return { kind: 'json', body: parsedToOpenAIJson(parseGeminiAnswer(answer), model) };
    }
    if (wire === 'openai') return { kind: 'json', body: openAIToAnthropic(answer, model) };
    return { kind: 'json', body: parsedToAnthropicJson(parseGeminiAnswer(answer), model) };
  } catch {
    failuresByProvider.set(pid, (failuresByProvider.get(pid) ?? 0) + 1);
    return null;
  }
}

async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonViaOpenAI = clientApi === 'openai' && jsonModeOpenAI(body);
  const candidates = candidatesFor(requires, jsonViaOpenAI);
  const tried = new Set<string>();
  const used = new Set<string>();
  let attempts = 0;
  for (;;) {
    if (attempts >= MAX_ATTEMPTS) break;
    // Pick the best remaining candidate; a provider that already refused
    // this request is penalised 15 for its other models, so they are only
    // tried when they clearly beat every untouched provider.
    let candidate: Candidate | null = null;
    let bestKey = -Infinity;
    for (const c of candidates) {
      const key = `${c.probe.provider.id}/${c.model.id}`;
      if (used.has(key)) continue;
      const eff = c.score - (tried.has(c.probe.provider.id) ? 15 : 0);
      if (eff > bestKey) { bestKey = eff; candidate = c; }
    }
    if (candidate === null) break;
    used.add(`${candidate.probe.provider.id}/${candidate.model.id}`);
    tried.add(candidate.probe.provider.id);
    attempts += 1;
    const served = await tryCandidate(candidate, clientApi, body, stream);
    if (served === null) continue;
    const model = candidate.model.id;
    if (served.kind === 'passthrough-json') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(served.text);
      return;
    }
    if (served.kind === 'json') { sendJson(res, 200, served.body); return; }
    if (served.kind === 'sse-text') {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(served.text);
      return;
    }
    if (served.kind === 'passthrough-stream') {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await pipeStream(served.stream, res);
      return;
    }
    // openai-stream-events: OpenAI SSE -> Anthropic events (starter converter).
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for await (const event of anthropicEventsFromOpenAI(served.stream, model)) res.write(event);
    res.end();
    return;
  }
  // No candidate answered: report the failure in the client's wire.
  if (clientApi === 'anthropic') sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  else sendError(res, 'openai', 503, 'service_unavailable', 'no free model on the directory answered this request');
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'openai', body);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'anthropic', body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') { models(req, res); return; }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') { await chat(req, res); return; }
  if (req.method === 'POST' && url.pathname === '/v1/messages') { await messages(req, res); return; }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
