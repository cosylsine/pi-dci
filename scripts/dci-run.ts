#!/usr/bin/env node
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Agent, type AgentEvent, type AgentTool, type StreamFn } from "@earendil-works/pi-agent-core";
import {
	createAssistantMessageEventStream,
	fauxAssistantMessage,
	fauxToolCall,
	type Api,
	type AssistantMessage,
	type Context,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type Tool,
	type ToolCall,
	Type,
} from "@earendil-works/pi-ai";
import { Value } from "typebox/value";

const DEFAULT_CONTRACT = {
	topK: 10,
	maxAgentSteps: 1200,
	maxFinalRepairTurns: 3,
	sampleTimeoutSeconds: 600,
	toolResultMaxChars: 20_000,
	maxTotalModelTokens: 128_000,
	maxFinalOutputTokens: 1_024,
	systemPrompt: `/no_think
You are the DCI-Bench retrieval agent.

Your task is to search the local corpus for documents relevant to the user's query.
The corpus is available as markdown files under /workspace/corpus. You may inspect
files with the provided read and bash tools. Do not use network access, web search,
external retrievers, embeddings, hidden labels, memory, skills, MCP tools, or
sub-agents.

You must inspect corpus document content before giving the final answer. A list
of filenames or file ids is not evidence. Use full-text search commands such as
rg -n -i or grep -Rin over important query keywords and synonyms, then read the
most promising document files.

Return only JSON matching this schema:
{"ranked_doc_ids":["doc_id_1","doc_id_2","doc_id_3"]}

Rules:
- Return at most 10 document ids.
- Rank the most relevant document first.
- Use the original document id, recovered from the corpus filename by removing
  the .md suffix and URL percent-decoding it when needed.
- Do not include explanations, markdown fences, scores, or extra keys.`,
};

type Args = {
	queryFile: string;
	workspace: string;
	output: string;
	trace?: string;
	modelProvider: "bridge" | "mock";
	mockScript?: string;
	openaiBaseUrl: string;
	openaiApiKey: string;
	model: string;
	maxAgentSteps: number;
};

type MockStep =
	| { type: "tool_call"; name: string; arguments: Record<string, unknown>; id?: string }
	| { type: "final"; text: string };

function usage(input = 0, output = 0) {
	return {
		input,
		output,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: input + output,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function textOf(content: Message["content"] | AssistantMessage["content"]): string {
	if (typeof content === "string") return content;
	return content
		.map((part) => {
			if (part.type === "text") return part.text;
			if (part.type === "toolCall") return `[toolCall:${part.name}]`;
			return "";
		})
		.filter(Boolean)
		.join("\n");
}

function parseArgs(argv: string[]): Args {
	const values = new Map<string, string | true>();
	for (let index = 0; index < argv.length; index++) {
		const arg = argv[index];
		if (!arg.startsWith("--")) throw new Error(`Unexpected positional argument: ${arg}`);
		const key = arg.slice(2);
		const next = argv[index + 1];
		if (!next || next.startsWith("--")) {
			values.set(key, true);
		} else {
			values.set(key, next);
			index++;
		}
	}
	const required = (key: string): string => {
		const value = values.get(key);
		if (typeof value !== "string" || !value) throw new Error(`Missing --${key}`);
		return value;
	};
	const positiveInt = (key: string, fallback: number): number => {
		const value = values.get(key);
		if (value === undefined) return fallback;
		if (typeof value !== "string") throw new Error(`--${key} requires an integer value`);
		const parsed = Number.parseInt(value, 10);
		if (!Number.isSafeInteger(parsed) || parsed < 1) throw new Error(`--${key} must be a positive integer`);
		return parsed;
	};
	return {
		queryFile: required("query-file"),
		workspace: required("workspace"),
		output: required("output"),
		trace: typeof values.get("trace") === "string" ? String(values.get("trace")) : undefined,
		modelProvider: values.get("mock-script") ? "mock" : "bridge",
		mockScript: typeof values.get("mock-script") === "string" ? String(values.get("mock-script")) : undefined,
		openaiBaseUrl: String(values.get("openai-base-url") ?? process.env.OPENAI_BASE_URL ?? "http://localhost:13131/v1"),
		openaiApiKey: String(values.get("openai-api-key") ?? process.env.OPENAI_API_KEY ?? "inspect"),
		model: String(values.get("model") ?? "inspect"),
		maxAgentSteps: positiveInt("max-agent-steps", DEFAULT_CONTRACT.maxAgentSteps),
	};
}

const readSchema = Type.Object(
	{
		path: Type.String({ description: "Path to read. Absolute paths and paths relative to workspace are accepted." }),
		offset: Type.Optional(Type.Number({ description: "Zero-based character offset." })),
		limit: Type.Optional(Type.Number({ description: "Maximum characters to return." })),
	},
	{ additionalProperties: false },
);

const bashSchema = Type.Object(
	{
		command: Type.String({ description: "Shell command to run in the workspace." }),
		timeout_seconds: Type.Optional(Type.Number({ description: "Timeout in seconds, max 60." })),
	},
	{ additionalProperties: false },
);

function truncate(text: string): string {
	if (text.length <= DEFAULT_CONTRACT.toolResultMaxChars) return text;
	return `${text.slice(0, DEFAULT_CONTRACT.toolResultMaxChars)}\n[truncated]`;
}

type EvidenceState = {
	bodyReadFiles: Set<string>;
	contentSearchCommands: string[];
};

function workspaceRelativePath(workspace: string, path: string): string {
	const workspaceRoot = resolve(workspace);
	const full = resolve(path);
	return full === workspaceRoot || !full.startsWith(`${workspaceRoot}/`) ? full : full.slice(workspaceRoot.length + 1);
}

function commandLooksLikeContentInspection(command: string): boolean {
	const touchesCorpus = /(^|[\s'"])(\/workspace\/corpus|corpus\/|\$DCI_CORPUS|DCI_CORPUS)\b/.test(command);
	if (!touchesCorpus) return false;
	return /(^|[\s|;&()])(rg|grep|ag|awk|sed|cat|head|tail)\b/.test(command);
}

function hasBodyEvidence(evidence: EvidenceState): boolean {
	return evidence.bodyReadFiles.size > 0 || evidence.contentSearchCommands.length > 0;
}

function stripModelNoise(text: string): string {
	return text
		.replace(/<think>[\s\S]*?<\/think>/g, "")
		.replace(/<<content-internal>[\s\S]*?<\/content-internal>>/g, "")
		.trim();
}

function parsePseudoToolCall(text: string): ToolCall | undefined {
	const cleaned = stripModelNoise(text);
	try {
		const parsed = JSON.parse(cleaned);
		if (
			parsed &&
			typeof parsed === "object" &&
			(parsed.name === "read" || parsed.name === "bash") &&
			parsed.arguments &&
			typeof parsed.arguments === "object"
		) {
			return fauxToolCall(parsed.name, parsed.arguments);
		}
	} catch {
		return undefined;
	}
	return undefined;
}

function resolveWorkspacePath(workspace: string, requested: string): string {
	const mapped = requested === "/workspace" || requested.startsWith("/workspace/")
		? `${workspace}${requested.slice("/workspace".length)}`
		: requested;
	const full = mapped.startsWith("/") ? resolve(mapped) : resolve(workspace, mapped);
	const workspaceRoot = resolve(workspace);
	if (full !== workspaceRoot && !full.startsWith(`${workspaceRoot}/`)) {
		throw new Error(`Path escapes workspace: ${requested}`);
	}
	return full;
}

function rewriteLocalSandboxCommand(command: string, workspace: string, scratch: string): string {
	return command
		.replace(/\/workspace/g, workspace)
		.replace(/\/scratch/g, scratch);
}

function createReadTool(workspace: string, evidence: EvidenceState): AgentTool<typeof readSchema> {
	return {
		name: "read",
		label: "read",
		description: "Read UTF-8 text from a file in the mounted workspace.",
		parameters: readSchema,
		executionMode: "sequential",
		prepareArguments: (args) => Value.Parse(readSchema, args),
		execute: async (_toolCallId, params) => {
			const path = resolveWorkspacePath(workspace, params.path);
			const info = await stat(path);
			if (info.isDirectory()) {
				const entries = (await readdir(path)).sort();
				const text = truncate(entries.join("\n"));
				return { content: [{ type: "text", text }], details: { path, entries: entries.length, bodyEvidence: false } };
			}
			const raw = await readFile(path, "utf8");
			const offset = Math.max(0, Math.floor(params.offset ?? 0));
			const limit = Math.max(1, Math.floor(params.limit ?? DEFAULT_CONTRACT.toolResultMaxChars));
			const text = truncate(raw.slice(offset, offset + limit));
			const relativePath = workspaceRelativePath(workspace, path);
			evidence.bodyReadFiles.add(relativePath);
			return { content: [{ type: "text", text }], details: { path, bytes: Buffer.byteLength(text), bodyEvidence: true } };
		},
	};
}

function createBashTool(workspace: string, scratch: string, evidence: EvidenceState): AgentTool<typeof bashSchema> {
	return {
		name: "bash",
		label: "bash",
		description: "Run a local shell command in the workspace without network access.",
		parameters: bashSchema,
		executionMode: "sequential",
		prepareArguments: (args) => Value.Parse(bashSchema, args),
		execute: async (_toolCallId, params, signal) => {
			const timeoutMs = Math.min(60, Math.max(1, Math.floor(params.timeout_seconds ?? 10))) * 1000;
			const controller = new AbortController();
			const timeout = setTimeout(() => controller.abort(), timeoutMs);
			const combined = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
			try {
				const proc = spawn("bash", ["-lc", rewriteLocalSandboxCommand(params.command, workspace, scratch)], {
					cwd: workspace,
					env: {
						...process.env,
						HOME: scratch,
						DCI_WORKSPACE: workspace,
						DCI_CORPUS: resolve(workspace, "corpus"),
						DCI_SCRATCH: scratch,
					},
					signal: combined,
				});
				const stdoutChunks: Buffer[] = [];
				const stderrChunks: Buffer[] = [];
				proc.stdout.on("data", (chunk) => stdoutChunks.push(Buffer.from(chunk)));
				proc.stderr.on("data", (chunk) => stderrChunks.push(Buffer.from(chunk)));
				const exitCode = await new Promise<number>((resolvePromise, reject) => {
					proc.on("error", reject);
					proc.on("close", (code) => resolvePromise(code ?? -1));
				});
				const stdout = Buffer.concat(stdoutChunks).toString("utf8");
				const stderr = Buffer.concat(stderrChunks).toString("utf8");
				const text = truncate([stdout, stderr ? `STDERR:\n${stderr}` : "", `exit_code=${exitCode}`].filter(Boolean).join("\n"));
				const bodyEvidence = exitCode === 0 && stdout.trim().length > 0 && commandLooksLikeContentInspection(params.command);
				if (bodyEvidence) evidence.contentSearchCommands.push(params.command);
				return { content: [{ type: "text", text }], details: { exitCode, bodyEvidence }, terminate: false };
			} finally {
				clearTimeout(timeout);
			}
		},
	};
}

function finalAnswerFrom(
	messages: readonly { role: string; content?: Message["content"] | AssistantMessage["content"] }[],
): string {
	for (const message of [...messages].reverse()) {
		if (message.role === "assistant" && message.content !== undefined) {
			const text = textOf(message.content).trim();
			if (text) return text;
		}
	}
	return "";
}

type ParsedFinal = {
	ranked_doc_ids: string[];
	adapter?: string;
};

function normalizeDocId(docId: string): string {
	return decodeURIComponent(docId.replace(/\.md$/i, ""));
}

function parseRankedIds(value: unknown): string[] | undefined {
	if (
		Array.isArray(value) &&
		value.length > 0 &&
		value.length <= DEFAULT_CONTRACT.topK &&
		value.every((item: unknown) => typeof item === "string" && item.length > 0)
	) {
		const ranked = value.map(normalizeDocId);
		if (new Set(ranked).size === ranked.length) return ranked;
	}
	return undefined;
}

function parseFinalJson(text: string): ParsedFinal | undefined {
	try {
		const parsed = JSON.parse(stripModelNoise(text));
		const strictRanked = parseRankedIds(parsed?.ranked_doc_ids);
		if (
			parsed &&
			typeof parsed === "object" &&
			Object.keys(parsed).length === 1 &&
			strictRanked
		) {
			return { ranked_doc_ids: strictRanked };
		}
		const topDocuments = parseRankedIds(parsed?.top_documents);
		if (
			parsed &&
			typeof parsed === "object" &&
			Object.keys(parsed).length === 1 &&
			topDocuments
		) {
			return { ranked_doc_ids: topDocuments, adapter: "top_documents_to_ranked_doc_ids" };
		}
	} catch {
		return undefined;
	}
	return undefined;
}

function finalAcceptanceIssue(parsed: ParsedFinal | undefined, evidence: EvidenceState): string | undefined {
	if (!parsed) return "final answer did not match DCI ranked_doc_ids JSON schema";
	if (!hasBodyEvidence(evidence)) {
		return "final answer was submitted before inspecting corpus document content";
	}
	return undefined;
}

function createModel(provider: Args["modelProvider"], id: string, baseUrl: string): Model<Api> {
	return {
		id,
		name: id,
		api: provider === "mock" ? "mock" : "openai-completions",
		provider: provider === "mock" ? "mock" : "inspect-bridge",
		baseUrl,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: DEFAULT_CONTRACT.maxTotalModelTokens,
		maxTokens: DEFAULT_CONTRACT.maxFinalOutputTokens,
	};
}

async function createMockStreamFn(mockScript: string): Promise<StreamFn> {
	const steps = JSON.parse(await readFile(mockScript, "utf8")) as MockStep[];
	let callIndex = 0;
	return async (model) => {
		const stream = createAssistantMessageEventStream();
		queueMicrotask(() => {
			const step = steps[callIndex++];
			if (!step) {
				const msg = fauxAssistantMessage("No mock response queued", { stopReason: "error" });
				stream.push({ type: "error", reason: "error", error: msg });
				stream.end(msg);
				return;
			}
			const message =
				step.type === "tool_call"
					? fauxAssistantMessage([fauxToolCall(step.name, step.arguments, { id: step.id })], { stopReason: "toolUse" })
					: fauxAssistantMessage(step.text, { stopReason: "stop" });
			message.provider = model.provider;
			message.model = model.id;
			message.api = model.api;
			stream.push({ type: "done", reason: message.stopReason === "toolUse" ? "toolUse" : "stop", message });
			stream.end(message);
		});
		return stream;
	};
}

function toOpenAIMessages(context: Context): Record<string, unknown>[] {
	const messages: Record<string, unknown>[] = [];
	if (context.systemPrompt) messages.push({ role: "system", content: context.systemPrompt });
	for (const message of context.messages) {
		if (message.role === "user") {
			messages.push({ role: "user", content: textOf(message.content) });
		} else if (message.role === "assistant") {
			const text = textOf(message.content).replace(/\[toolCall:[^\]]+\]/g, "").trim();
			const toolCalls = message.content
				.filter((part): part is ToolCall => part.type === "toolCall")
				.map((part) => ({
					id: part.id,
					type: "function",
					function: { name: part.name, arguments: JSON.stringify(part.arguments ?? {}) },
				}));
			messages.push({
				role: "assistant",
				content: text || null,
				...(toolCalls.length ? { tool_calls: toolCalls } : {}),
			});
		} else if (message.role === "toolResult") {
			messages.push({
				role: "tool",
				tool_call_id: message.toolCallId,
				name: message.toolName,
				content: textOf(message.content),
			});
		}
	}
	return messages;
}

function toOpenAITools(tools: readonly Tool[] | undefined): Record<string, unknown>[] {
	return (tools ?? []).map((tool) => ({
		type: "function",
		function: {
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters,
		},
	}));
}

function createBridgeStreamFn(args: Args): StreamFn {
	return async (model, context, options?: SimpleStreamOptions) => {
		const stream = createAssistantMessageEventStream();
		queueMicrotask(async () => {
			const baseUrl = args.openaiBaseUrl.replace(/\/+$/, "");
			const request = {
				model: args.model,
				messages: toOpenAIMessages(context),
				tools: toOpenAITools(context.tools),
				tool_choice: options?.toolChoice ?? "auto",
				max_tokens: DEFAULT_CONTRACT.maxFinalOutputTokens,
				temperature: 0,
			};
			try {
				const response = await fetch(`${baseUrl}/chat/completions`, {
					method: "POST",
					headers: {
						"content-type": "application/json",
						authorization: `Bearer ${args.openaiApiKey}`,
					},
					body: JSON.stringify(request),
				});
				const payload = await response.json();
				if (!response.ok) throw new Error(JSON.stringify(payload));
				const choice = payload.choices?.[0];
				const output = choice?.message ?? {};
				const content: AssistantMessage["content"] = [];
				if (typeof output.content === "string" && output.content) {
					content.push({ type: "text", text: output.content });
				}
				for (const toolCall of output.tool_calls ?? []) {
					const rawArgs = toolCall.function?.arguments ?? "{}";
					let parsedArgs: Record<string, unknown> = {};
					try {
						parsedArgs = JSON.parse(rawArgs);
					} catch {
						parsedArgs = { raw_arguments: rawArgs };
					}
					content.push({
						type: "toolCall",
						id: toolCall.id ?? `tool_${createHash("sha1").update(JSON.stringify(toolCall)).digest("hex").slice(0, 12)}`,
						name: toolCall.function?.name,
						arguments: parsedArgs,
					});
				}
				const pseudoToolCall = content.length === 1 && content[0]?.type === "text" ? parsePseudoToolCall(content[0].text) : undefined;
				if (pseudoToolCall) {
					content.splice(0, content.length, pseudoToolCall);
				}
				const stopReason = content.some((part) => part.type === "toolCall") ? "toolUse" : "stop";
				const message: AssistantMessage = {
					role: "assistant",
					content,
					api: model.api,
					provider: model.provider,
					model: model.id,
					usage: usage(payload.usage?.prompt_tokens ?? 0, payload.usage?.completion_tokens ?? 0),
					stopReason,
					rawStopReason: choice?.finish_reason,
					timestamp: Date.now(),
				};
				stream.push({ type: "done", reason: stopReason, message });
				stream.end(message);
			} catch (error) {
				const message: AssistantMessage = {
					role: "assistant",
					content: [{ type: "text", text: "" }],
					api: model.api,
					provider: model.provider,
					model: model.id,
					usage: usage(),
					stopReason: "error",
					errorMessage: error instanceof Error ? error.message : String(error),
					timestamp: Date.now(),
				};
				stream.push({ type: "error", reason: "error", error: message });
				stream.end(message);
			}
		});
		return stream;
	};
}

function buildPrompt(query: string): string {
	return `/no_think
Query:
${query}

Use bash or read to inspect /workspace/corpus before answering. You must inspect
document content before giving the final answer; a filename listing alone is not
evidence. Prefer full-text searches such as rg -n -i over query keywords and
synonyms, then read promising corpus files to verify relevance. Do not infer
document ids from ls/find ordering.

Return the top relevant document ids as JSON only, with exactly this key:
ranked_doc_ids. Do not use top_documents, doc_ids, scores, explanations, or .md
filename suffixes.`;
}

function buildRepairPrompt(reason: string): string {
	return `/no_think
The previous answer cannot be accepted: ${reason}.

Continue the same retrieval task. Use full-text inspection of /workspace/corpus
now: run rg/grep over important query keywords and synonyms, read the most
promising document files, then return only valid JSON with ranked_doc_ids. Do
not guess from filename lists or return an empty list unless you have exhausted
reasonable content searches.`;
}

async function main() {
	const args = parseArgs(process.argv.slice(2));
	const workspace = resolve(args.workspace);
	const scratch = resolve(workspace, "..", "scratch");
	const output = resolve(args.output);
	const trace = args.trace ? resolve(args.trace) : undefined;
	const query = await readFile(resolve(args.queryFile), "utf8");
	await mkdir(dirname(output), { recursive: true });
	await mkdir(scratch, { recursive: true });
	if (trace) await mkdir(dirname(trace), { recursive: true });

	let turns = 0;
	const events: unknown[] = [];
	const evidence: EvidenceState = { bodyReadFiles: new Set(), contentSearchCommands: [] };
	const repairReasons: string[] = [];
	const model = createModel(args.modelProvider, args.model, args.openaiBaseUrl);
	const streamFn = args.modelProvider === "mock" ? await createMockStreamFn(args.mockScript!) : createBridgeStreamFn(args);
	const agent = new Agent({
		streamFn,
		initialState: {
				systemPrompt: DEFAULT_CONTRACT.systemPrompt,
				model,
				thinkingLevel: "off",
				tools: [createReadTool(workspace, evidence), createBashTool(workspace, scratch, evidence)],
			},
		toolExecution: "sequential",
		shouldStopAfterTurn: async ({ context }) => {
			turns++;
			const parsed = parseFinalJson(finalAnswerFrom(context.messages));
			if (parsed && hasBodyEvidence(evidence)) return true;
			return turns >= args.maxAgentSteps;
		},
	});
	agent.subscribe((event: AgentEvent) => {
		if (event.type === "turn_end" || event.type === "tool_execution_start" || event.type === "tool_execution_end") {
			events.push(event);
		}
	});
	await agent.prompt(buildPrompt(query));
	let finalText = finalAnswerFrom(agent.state.messages);
	let parsed = parseFinalJson(finalText);
	let finalIssue = finalAcceptanceIssue(parsed, evidence);
	while (finalIssue && turns < args.maxAgentSteps && repairReasons.length < DEFAULT_CONTRACT.maxFinalRepairTurns) {
		repairReasons.push(finalIssue);
		await agent.prompt(buildRepairPrompt(finalIssue));
		finalText = finalAnswerFrom(agent.state.messages);
		parsed = parseFinalJson(finalText);
		finalIssue = finalAcceptanceIssue(parsed, evidence);
	}
	const acceptedParsed = parsed !== undefined && finalIssue === undefined ? parsed : undefined;
	const toolCalls = agent.state.messages.filter((message) => message.role === "toolResult").length;
	const result = {
		valid_output: acceptedParsed !== undefined,
		failure_reason: acceptedParsed ? null : (finalIssue ?? "max_agent_steps reached before an acceptable final answer"),
		ranked_doc_ids: acceptedParsed?.ranked_doc_ids ?? [],
		raw_final_text: finalText,
		agent_steps: turns,
		tool_calls: toolCalls,
		final_output_adapter: acceptedParsed?.adapter ?? null,
		repair_attempts: repairReasons.length,
		repair_reasons: repairReasons,
		body_evidence: {
			read_files: [...evidence.bodyReadFiles].sort(),
			content_search_commands: evidence.contentSearchCommands,
		},
		model_provider: model.provider,
		model: model.id,
		contract: {
			top_k: DEFAULT_CONTRACT.topK,
			max_agent_steps: args.maxAgentSteps,
			tool_allowlist: ["read", "bash"],
		},
	};
	await writeFile(output, `${JSON.stringify(result, null, 2)}\n`, "utf8");
	if (trace) {
		await writeFile(
			trace,
			`${JSON.stringify({ argv: process.argv.slice(2), cwd: process.cwd(), workspace, output, events, messages: agent.state.messages }, null, 2)}\n`,
			"utf8",
		);
	}
	if (!acceptedParsed) {
		console.error(`Invalid DCI final answer from ${basename(fileURLToPath(import.meta.url))}: ${finalText}`);
		process.exitCode = 2;
	}
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});
