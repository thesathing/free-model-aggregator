// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/* ---- Added: cross-wire builders and converters (Gemini / Anthropic upstream) ---- */

export function stopReasonAnthropicOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}
export function finishReasonOpenAIOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Split text into SSE-sized pieces (provider uses 13 chars). */
export function splitPieces(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += 13) out.push(text.slice(at, at + 13));
  return out.length === 0 ? [''] : out;
}

/** Parse SSE text into frames {event, data}. */
export function parseSse(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function openAITextParts(messages: Json[]): { system: string; parts: Json[] } {
  // Collect system text and build Gemini-style content parts from OpenAI messages.
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(messages) ? messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') { const t = textOf(content); if (t !== '') systemParts.push(t); continue; }
    const parts: Json[] = [];
    if (typeof content === 'string') { if (content !== '') parts.push({ text: content }); }
    else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
          if (m) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
          else parts.push({ fileData: { fileUri: url } });
        }
      }
    }
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        try { const p: unknown = JSON.parse(typeof call.function.arguments === 'string' ? call.function.arguments : '{}'); if (isRecord(p)) args = p; } catch { /* keep {} */ }
        parts.push({ functionCall: { name: call.function.name, args } });
      }
    }
    if (role === 'tool') {
      contents.push({ role: 'user', parts: [{ functionResponse: { name: typeof raw.name === 'string' ? raw.name : 'tool', response: { content: typeof content === 'string' ? content : textOf(content) } } }] });
      continue;
    }
    if (parts.length > 0) contents.push({ role: role === 'assistant' ? 'model' : 'user', parts });
  }
  return { system: systemParts.join('\n'), parts: contents };
}

function anthropicTextParts(body: Json): { system: string; parts: Json[] } {
  const system = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') { if (content !== '') parts.push({ text: content }); }
    else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          else if (block.source.type === 'url' && typeof block.source.url === 'string') parts.push({ fileData: { fileUri: block.source.url } });
        } else if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        } else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : 'tool', response: { content: toolResultText(block.content) } } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  return { system, parts: contents };
}

function geminiToolsFromOpenAI(body: Json): Json[] {
  const decls: Json[] = [];
  if (Array.isArray(body.tools)) for (const raw of body.tools) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    decls.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
  }
  return decls;
}
function geminiToolsFromAnthropic(body: Json): Json[] {
  const decls: Json[] = [];
  if (Array.isArray(body.tools)) for (const raw of body.tools) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    decls.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
  }
  return decls;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGemini(body: Json, model: string): Json {
  const { system, parts } = openAITextParts(Array.isArray(body.messages) ? body.messages : []);
  const out: Json = { contents: parts };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const decls = geminiToolsFromOpenAI(body);
  if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  const config: Json = {};
  const maxTok = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTok !== undefined) config.maxOutputTokens = maxTok;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops: string[] = [];
  if (typeof body.stop === 'string') stops.push(body.stop);
  else if (Array.isArray(body.stop)) for (const s of body.stop) if (typeof s === 'string') stops.push(s);
  if (stops.length > 0) config.stopSequences = stops;
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (fmt === 'json_object' || fmt === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGemini(body: Json, _model: string): Json {
  const { system, parts } = anthropicTextParts(body);
  const out: Json = { contents: parts };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const decls = geminiToolsFromAnthropic(body);
  if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

export interface ToolCallOut { id: string; name: string; arguments: string; }
export interface ParsedAnswer {
  text: string;
  toolCalls: ToolCallOut[];
  stop: string; // provider-native stop string
  inputTokens: number;
  outputTokens: number;
}

/** A Gemini JSON answer parsed into text / tool calls / stop / usage. */
export function parseGeminiAnswer(body: Json): ParsedAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: ToolCallOut[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) toolCalls.push({ id: `toolu_${toolCalls.length}`, name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) });
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    toolCalls,
    stop: typeof cand.finishReason === 'string' ? cand.finishReason : 'STOP',
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
  };
}

/** A Gemini SSE stream parsed into one answer (buffered). */
export function parseGeminiStream(text: string): ParsedAnswer {
  let outText = '';
  const toolCalls: ToolCallOut[] = [];
  let stop = 'STOP';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of parseSse(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    let chunk: unknown;
    try { chunk = JSON.parse(frame.data); } catch { continue; }
    if (!isRecord(chunk)) continue;
    const parsed = parseGeminiAnswer(chunk);
    outText += parsed.text;
    for (const tc of parsed.toolCalls) toolCalls.push(tc);
    if (parsed.stop !== '') stop = parsed.stop;
    if (isRecord(chunk.usageMetadata)) { inputTokens = numberOrZero(chunk.usageMetadata.promptTokenCount); outputTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount); }
  }
  return { text: outText, toolCalls, stop, inputTokens, outputTokens };
}

/** An Anthropic JSON answer parsed into text / tool calls / stop / usage. */
export function parseAnthropicAnswer(body: Json): ParsedAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: ToolCallOut[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') toolCalls.push({ id: typeof block.id === 'string' ? block.id : 'toolu_unknown', name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, toolCalls, stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn', inputTokens: numberOrZero(usage.input_tokens), outputTokens: numberOrZero(usage.output_tokens) };
}

/** An Anthropic SSE stream parsed into one answer (buffered). */
export function parseAnthropicStream(text: string): ParsedAnswer {
  let outText = '';
  const toolMap = new Map<number, ToolCallOut>();
  let stop = 'end_turn';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of parseSse(text)) {
    let data: unknown;
    try { data = JSON.parse(frame.data); } catch { continue; }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
    else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const idx = typeof data.index === 'number' ? data.index : toolMap.size;
      toolMap.set(idx, { id: typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${idx}`, name: typeof data.content_block.name === 'string' ? data.content_block.name : '', arguments: '' });
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const idx = typeof data.index === 'number' ? data.index : 0;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') outText += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const tc = toolMap.get(idx);
        if (tc) tc.arguments += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: outText, toolCalls: [...toolMap.values()], stop, inputTokens, outputTokens };
}

/** An OpenAI JSON answer parsed into text / tool calls / stop / usage. */
export function parseOpenAIAnswer(body: Json): ParsedAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: ToolCallOut[] = [];
  if (Array.isArray(message.tool_calls)) for (const call of message.tool_calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({ id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}) });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, toolCalls, stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop', inputTokens: numberOrZero(usage.prompt_tokens), outputTokens: numberOrZero(usage.completion_tokens) };
}

/** An OpenAI SSE stream parsed into one answer (buffered). */
export function parseOpenAIStream(text: string): ParsedAnswer {
  let outText = '';
  const toolMap = new Map<number, ToolCallOut>();
  let stop = 'stop';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of parseSse(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try { chunk = JSON.parse(payload); } catch { continue; }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) { inputTokens = numberOrZero(chunk.usage.prompt_tokens); outputTokens = numberOrZero(chunk.usage.completion_tokens); }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') outText += delta.content;
    if (Array.isArray(delta.tool_calls)) for (const call of delta.tool_calls) {
      if (!isRecord(call)) continue;
      const idx = typeof call.index === 'number' ? call.index : toolMap.size;
      let tc = toolMap.get(idx);
      if (!tc) { tc = { id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${idx}`, name: '', arguments: '' }; toolMap.set(idx, tc); }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') tc.name = fn.name;
      if (typeof call.id === 'string' && call.id !== '') tc.id = call.id;
      if (typeof fn.arguments === 'string') tc.arguments += fn.arguments;
    }
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: outText, toolCalls: [...toolMap.values()], stop, inputTokens, outputTokens };
}

/** Build an OpenAI JSON completion from a parsed answer. */
export function parsedToOpenAIJson(ans: ParsedAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: ans.text };
  if (ans.toolCalls.length > 0) message.tool_calls = ans.toolCalls.map((tc) => ({ id: tc.id, type: 'function', function: { name: tc.name, arguments: tc.arguments } }));
  let finish = 'stop';
  if (ans.stop === 'max_tokens' || ans.stop === 'MAX_TOKENS' || ans.stop === 'length') finish = 'length';
  else if (ans.stop === 'tool_use' || ans.stop === 'tool_calls') finish = 'tool_calls';
  else if (ans.stop === 'refusal' || ans.stop === 'SAFETY' || ans.stop === 'content_filter') finish = 'content_filter';
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finish, logprobs: null }],
    usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens },
  };
}

/** Build an Anthropic JSON message from a parsed answer. */
export function parsedToAnthropicJson(ans: ParsedAnswer, model: string): Json {
  const content: Json[] = [];
  if (ans.text !== '') content.push({ type: 'text', text: ans.text });
  for (const tc of ans.toolCalls) content.push({ type: 'tool_use', id: tc.id, name: tc.name, input: argumentsOf(tc.arguments) });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonAnthropicOf(ans.stop === 'STOP' ? 'stop' : ans.stop),
    stop_sequence: null,
    usage: { input_tokens: ans.inputTokens, output_tokens: ans.outputTokens },
  };
}

/** Build an OpenAI SSE stream from a parsed answer. */
export function parsedToOpenAIStream(ans: ParsedAnswer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitPieces(ans.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  for (let i = 0; i < ans.toolCalls.length; i++) {
    const tc = ans.toolCalls[i];
    frames.push(sseEvent(null, chunk({ tool_calls: [{ index: i, id: tc.id, type: 'function', function: { name: tc.name, arguments: tc.arguments } }] }, null)));
  }
  let finish = 'stop';
  if (ans.stop === 'max_tokens' || ans.stop === 'MAX_TOKENS' || ans.stop === 'length') finish = 'length';
  else if (ans.stop === 'tool_use' || ans.stop === 'tool_calls') finish = 'tool_calls';
  else if (ans.stop === 'refusal' || ans.stop === 'SAFETY' || ans.stop === 'content_filter') finish = 'content_filter';
  frames.push(sseEvent(null, chunk({}, finish)));
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens } }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** Build an Anthropic SSE stream from a parsed answer. */
export function parsedToAnthropicStream(ans: ParsedAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(sseEvent('message_start', { type: 'message_start', message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: ans.inputTokens, output_tokens: 0 } } }));
  let index = 0;
  if (ans.text !== '' || ans.toolCalls.length === 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitPieces(ans.text)) frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const tc of ans.toolCalls) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: tc.id, name: tc.name, input: {} } }));
    if (tc.arguments !== '') frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: tc.arguments } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReasonAnthropicOf(ans.stop === 'STOP' ? 'stop' : ans.stop), stop_sequence: null }, usage: { output_tokens: ans.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** An OpenAI Chat Completions request as an Anthropic Messages request. */
export function openAIToAnthropicReq(body: Json, model: string): Json {
  const { system } = openAITextParts(Array.isArray(body.messages) ? body.messages : []);
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const t = textOf(content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      if (Array.isArray(raw.tool_calls)) for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof content === 'string' ? content : textOf(content) }] });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') parts.push({ type: 'text', text: content });
    else if (Array.isArray(content)) for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url = block.image_url.url;
        const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
        if (m) parts.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
        else parts.push({ type: 'image', source: { type: 'url', url } });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    messages.push({ role: 'user', content: parts.length === 1 && parts[0].type === 'text' ? (parts[0] as Json).text : parts });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTok = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  out.max_tokens = maxTok !== undefined ? maxTok : 1024;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  if (typeof body.tool_choice === 'string') {
    if (body.tool_choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (body.tool_choice === 'none') out.tool_choice = { type: 'none' };
    else if (body.tool_choice === 'required') out.tool_choice = { type: 'any' };
  } else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: body.tool_choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}
