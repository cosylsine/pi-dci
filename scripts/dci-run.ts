#!/usr/bin/env node
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { constants as fsConstants } from "node:fs";
import { access, lstat, mkdir, mkdtemp, readFile, readdir, realpath, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve, sep } from "node:path";
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
	maxScratchBytes: 20_000_000,
	maxTotalModelTokens: 128_000,
	maxFinalOutputTokens: 1_024,
	maxCorpusDocumentBytes: 1_000_000,
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
	preflightOnly: boolean;
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
	const preflightOnly = values.has("preflight-only");
	return {
		preflightOnly,
		queryFile: preflightOnly ? "" : required("query-file"),
		workspace: required("workspace"),
		output: preflightOnly ? "" : required("output"),
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

async function directoryUsageBytes(root: string): Promise<number> {
	let total = 0;
	const pending = [root];
	while (pending.length > 0) {
		const directory = pending.pop();
		if (!directory) continue;
		let entries;
		try {
			entries = await readdir(directory, { withFileTypes: true });
		} catch (error) {
			const code = error instanceof Error && "code" in error ? error.code : undefined;
			if (code === "ENOENT") continue;
			throw error;
		}
		for (const entry of entries) {
			const path = join(directory, entry.name);
			let info;
			try {
				info = await lstat(path);
			} catch (error) {
				const code = error instanceof Error && "code" in error ? error.code : undefined;
				if (code === "ENOENT") continue;
				throw error;
			}
			total += Math.max(info.size, info.blocks * 512);
			if (total > DEFAULT_CONTRACT.maxScratchBytes) return total;
			if (info.isDirectory() && !info.isSymbolicLink()) pending.push(path);
		}
	}
	return total;
}

async function findExecutable(name: string): Promise<string | undefined> {
	for (const directory of (process.env.PATH ?? "/usr/bin:/bin").split(":")) {
		if (!directory) continue;
		const candidate = join(directory, name);
		try {
			await access(candidate, fsConstants.X_OK);
			return await realpath(candidate);
		} catch {
			// Try the next PATH entry.
		}
	}
	return undefined;
}

async function validateTrustedExecutable(path: string, label: string): Promise<string> {
	const canonical = await realpath(path);
	const info = await stat(canonical);
	if (!info.isFile() || info.uid !== 0 || (info.mode & 0o022) !== 0) {
		throw new Error(`${label} must resolve to a root-owned executable that is not group/world writable: ${canonical}`);
	}
	await access(canonical, fsConstants.X_OK);
	return canonical;
}

type CorpusSandbox = {
	workspace: string;
	corpus: string;
	bwrap: string;
	rg?: string;
};

async function validateCorpusSandbox(workspace: string): Promise<CorpusSandbox> {
	const resolvedWorkspace = await realpath(workspace);
	const corpus = resolve(resolvedWorkspace, "corpus");
	const corpusInfo = await lstat(corpus);
	if (corpusInfo.isSymbolicLink() || !corpusInfo.isDirectory()) {
		throw new Error(`Corpus must be a real directory: ${corpus}`);
	}
	for (const entry of await readdir(corpus, { withFileTypes: true })) {
		if (entry.isSymbolicLink() || !entry.isFile()) {
			throw new Error(`Corpus entries must be regular files, found: ${entry.name}`);
		}
		if (!entry.name.endsWith(".md")) {
			throw new Error(`Corpus entries must be markdown files, found: ${entry.name}`);
		}
		const entryInfo = await lstat(join(corpus, entry.name));
		if (entryInfo.nlink !== 1) {
			throw new Error(`Corpus entries must not be hard links, found: ${entry.name}`);
		}
		if (entryInfo.size > DEFAULT_CONTRACT.maxCorpusDocumentBytes) {
			throw new Error(`Corpus document exceeds the size limit: ${entry.name}`);
		}
	}
	try {
		await access("/usr/bin/bwrap", fsConstants.X_OK);
	} catch {
		throw new Error("bubblewrap is required for corpus isolation; install it with: apt install -y bubblewrap");
	}
	const bwrap = await validateTrustedExecutable("/usr/bin/bwrap", "bubblewrap");
	if (bwrap !== "/usr/bin/bwrap") {
		throw new Error(`bubblewrap must resolve to the expected system path: ${bwrap}`);
	}
	const rgCandidate = await findExecutable("rg");
	let rg: string | undefined;
	if (rgCandidate) {
		try {
			rg = await validateTrustedExecutable(rgCandidate, "ripgrep");
		} catch {
			// grep remains available from the read-only system mount.
		}
	}
	return { workspace: resolvedWorkspace, corpus: await realpath(corpus), bwrap, rg };
}

function sandboxArgs(sandbox: CorpusSandbox, scratch: string, command: string): string[] {
	const args = [
		"--unshare-user",
		"--uid",
		"65534",
		"--gid",
		"65534",
		"--unshare-pid",
		"--unshare-net",
		"--unshare-ipc",
		"--unshare-uts",
		"--new-session",
		"--die-with-parent",
		"--cap-drop",
		"ALL",
		"--hostname",
		"dci-sample",
		"--dir",
		"/usr",
		"--ro-bind",
		"/usr/bin",
		"/usr/bin",
		"--ro-bind",
		"/usr/lib",
		"/usr/lib",
		"--ro-bind",
		"/usr/lib64",
		"/usr/lib64",
		"--symlink",
		"usr/bin",
		"/bin",
		"--symlink",
		"usr/lib",
		"/lib",
		"--symlink",
		"usr/lib64",
		"/lib64",
		"--dev",
		"/dev",
		"--dir",
		"/workspace",
		"--ro-bind",
		sandbox.corpus,
		"/workspace/corpus",
		"--bind",
		scratch,
		"/scratch",
		"--dir",
		"/sandbox-bin",
	];
	if (sandbox.rg) args.push("--ro-bind", sandbox.rg, "/sandbox-bin/rg");
	args.push(
		"--chdir",
		"/workspace",
		"--setenv",
		"PATH",
		"/sandbox-bin:/usr/bin:/bin",
		"--setenv",
		"HOME",
		"/scratch",
		"--setenv",
		"DCI_WORKSPACE",
		"/workspace",
		"--setenv",
		"DCI_CORPUS",
		"/workspace/corpus",
		"--setenv",
		"DCI_SCRATCH",
		"/scratch",
		"--setenv",
		"TMPDIR",
		"/scratch/tmp",
		"/bin/bash",
		"--noprofile",
		"--norc",
		"-c",
		"umask 077 && ulimit -S -n 256 && ulimit -H -n 256 && ulimit -S -u 128 && ulimit -H -u 128 && ulimit -S -f 2048 && ulimit -H -f 2048 && ulimit -S -v 1048576 && ulimit -H -v 1048576 && ulimit -S -t 60 && ulimit -H -t 60 && exec /bin/bash --noprofile --norc -c \"$1\"",
		"dci-sandbox",
		command,
	);
	return args;
}

async function runSandboxPreflight(sandbox: CorpusSandbox): Promise<void> {
	const scratch = await mkdtemp(join(tmpdir(), "dci-bench-preflight-"));
	try {
		await mkdir(join(scratch, "tmp"), { mode: 0o700 });
		const command =
			"test -r /workspace/corpus && test ! -e /mnt && test ! -e /root && test ! -e /proc && test ! -e /tmp && test -w /scratch && test -w \"$TMPDIR\"";
		const result = await new Promise<{ code: number; stderr: string }>((resolvePromise, reject) => {
			const proc = spawn(sandbox.bwrap, sandboxArgs(sandbox, scratch, command), {
				env: { PATH: "/usr/bin:/bin", LANG: "C.UTF-8", LC_ALL: "C.UTF-8" },
				stdio: ["ignore", "ignore", "pipe"],
			});
			const stderr: Buffer[] = [];
			proc.stderr.on("data", (chunk) => stderr.push(Buffer.from(chunk)));
			proc.on("error", reject);
			proc.on("close", (code) => resolvePromise({ code: code ?? -1, stderr: Buffer.concat(stderr).toString("utf8") }));
		});
		if (result.code !== 0) {
			throw new Error(`bubblewrap corpus sandbox preflight failed: ${result.stderr.trim() || `exit ${result.code}`}`);
		}
	} finally {
		await rm(scratch, { recursive: true, force: true });
	}
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

async function resolveCorpusPath(sandbox: CorpusSandbox, requested: string): Promise<string> {
	const mapped = requested === "/workspace/corpus" || requested.startsWith("/workspace/corpus/")
		? `${sandbox.corpus}${requested.slice("/workspace/corpus".length)}`
		: requested;
	const full = mapped.startsWith("/") ? resolve(mapped) : resolve(sandbox.workspace, mapped);
	if (full !== sandbox.corpus && !full.startsWith(`${sandbox.corpus}${sep}`)) {
		throw new Error(`Path escapes corpus: ${requested}`);
	}
	const info = await lstat(full);
	if (info.isSymbolicLink()) throw new Error(`Symlinks are not allowed in corpus reads: ${requested}`);
	const canonical = await realpath(full);
	if (canonical !== sandbox.corpus && !canonical.startsWith(`${sandbox.corpus}${sep}`)) {
		throw new Error(`Path escapes corpus after symlink resolution: ${requested}`);
	}
	return canonical;
}

function createReadTool(sandbox: CorpusSandbox, evidence: EvidenceState): AgentTool<typeof readSchema> {
	return {
		name: "read",
		label: "read",
		description: "Read UTF-8 text from a file in the mounted workspace.",
		parameters: readSchema,
		executionMode: "sequential",
		prepareArguments: (args) => Value.Parse(readSchema, args),
		execute: async (_toolCallId, params) => {
			const path = await resolveCorpusPath(sandbox, params.path);
			const info = await stat(path);
			if (info.isDirectory()) {
				const entries = (await readdir(path)).sort();
				const text = truncate(entries.join("\n"));
				return {
					content: [{ type: "text", text }],
					details: { path: "/workspace/corpus", entries: entries.length, bodyEvidence: false },
				};
			}
			if (!info.isFile()) throw new Error(`Corpus read requires a regular file: ${params.path}`);
			if (info.nlink !== 1) throw new Error(`Corpus read rejects hard-linked files: ${params.path}`);
			if (info.size > DEFAULT_CONTRACT.maxCorpusDocumentBytes) {
				throw new Error(`Corpus document exceeds the size limit: ${params.path}`);
			}
			const raw = await readFile(path, "utf8");
			const offset = Math.max(0, Math.floor(params.offset ?? 0));
			const limit = Math.max(1, Math.floor(params.limit ?? DEFAULT_CONTRACT.toolResultMaxChars));
			const text = truncate(raw.slice(offset, offset + limit));
			const relativePath = workspaceRelativePath(sandbox.workspace, path);
			evidence.bodyReadFiles.add(relativePath);
			return {
				content: [{ type: "text", text }],
				details: { path: `/workspace/${relativePath}`, bytes: Buffer.byteLength(text), bodyEvidence: true },
			};
		},
	};
}

function createBashTool(sandbox: CorpusSandbox, scratch: string, evidence: EvidenceState): AgentTool<typeof bashSchema> {
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
				const proc = spawn(sandbox.bwrap, sandboxArgs(sandbox, scratch, params.command), {
					env: { PATH: "/usr/bin:/bin", LANG: "C.UTF-8", LC_ALL: "C.UTF-8" },
					detached: true,
				});
				const stdoutChunks: Buffer[] = [];
				const stderrChunks: Buffer[] = [];
				let capturedBytes = 0;
				let outputLimitExceeded = false;
				let scratchLimitExceeded = false;
				let terminationRequested = false;
				const terminateProcessGroup = () => {
					if (terminationRequested) return;
					terminationRequested = true;
					try {
						if (proc.pid !== undefined) process.kill(-proc.pid, "SIGKILL");
						else proc.kill("SIGKILL");
					} catch {
						proc.kill("SIGKILL");
					}
				};
				const appendBounded = (chunks: Buffer[], chunk: Buffer): void => {
					const remaining = DEFAULT_CONTRACT.toolResultMaxChars - capturedBytes;
					if (remaining > 0) chunks.push(chunk.subarray(0, remaining));
					capturedBytes += chunk.length;
					if (capturedBytes > DEFAULT_CONTRACT.toolResultMaxChars && !outputLimitExceeded) {
						outputLimitExceeded = true;
						terminateProcessGroup();
					}
				};
				proc.stdout.on("data", (chunk) => {
					appendBounded(stdoutChunks, Buffer.from(chunk));
				});
				proc.stderr.on("data", (chunk) => {
					appendBounded(stderrChunks, Buffer.from(chunk));
				});
				let scratchCheck: Promise<void> | undefined;
				let scratchMonitorError: unknown;
				const checkScratchLimit = async () => {
					if ((await directoryUsageBytes(scratch)) > DEFAULT_CONTRACT.maxScratchBytes) {
						scratchLimitExceeded = true;
						terminateProcessGroup();
					}
				};
				const scratchMonitor = setInterval(() => {
					if (scratchCheck) return;
					scratchCheck = checkScratchLimit()
						.catch((error: unknown) => {
							scratchMonitorError = error;
							terminateProcessGroup();
						})
						.finally(() => {
							scratchCheck = undefined;
						});
				}, 100);
				combined.addEventListener("abort", terminateProcessGroup, { once: true });
				if (combined.aborted) terminateProcessGroup();
				let exitCode: number;
				try {
					exitCode = await new Promise<number>((resolvePromise, reject) => {
						proc.on("error", reject);
						proc.on("close", (code) => resolvePromise(code ?? -1));
					});
				} finally {
					clearInterval(scratchMonitor);
					combined.removeEventListener("abort", terminateProcessGroup);
					if (scratchCheck) await scratchCheck;
				}
				if (scratchMonitorError) throw scratchMonitorError;
				await checkScratchLimit();
				const stdout = Buffer.concat(stdoutChunks).toString("utf8");
				const stderr = Buffer.concat(stderrChunks).toString("utf8");
				const terminationReason = outputLimitExceeded
					? "[output limit exceeded; process terminated]"
					: scratchLimitExceeded
						? "[scratch limit exceeded; process terminated]"
					: controller.signal.aborted
						? "[timeout exceeded; process terminated]"
						: signal?.aborted
							? "[aborted; process terminated]"
							: "";
				const commandOutput = truncate([stdout, stderr ? `STDERR:\n${stderr}` : ""].filter(Boolean).join("\n"));
				const text = [commandOutput, terminationReason, `exit_code=${exitCode}`].filter(Boolean).join("\n");
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

function redactArgv(argv: string[]): string[] {
	const secretFlags = new Set(["--openai-api-key", "--api-key", "--token"]);
	let redactNext = false;
	return argv.map((arg) => {
		if (redactNext) {
			redactNext = false;
			return "[redacted]";
		}
		if (secretFlags.has(arg)) {
			redactNext = true;
			return arg;
		}
		const matchingFlag = [...secretFlags].find((flag) => arg.startsWith(`${flag}=`));
		return matchingFlag ? `${matchingFlag}=[redacted]` : arg;
	});
}

async function main() {
	const args = parseArgs(process.argv.slice(2));
	const sandbox = await validateCorpusSandbox(resolve(args.workspace));
	await runSandboxPreflight(sandbox);
	if (args.preflightOnly) {
		console.log(`bubblewrap corpus sandbox preflight passed: ${sandbox.corpus}`);
		return;
	}

	const scratch = await mkdtemp(join(tmpdir(), "dci-bench-sample-"));
	const output = resolve(args.output);
	const trace = args.trace ? resolve(args.trace) : undefined;
	try {
		await mkdir(join(scratch, "tmp"), { mode: 0o700 });
		const query = await readFile(resolve(args.queryFile), "utf8");
		await mkdir(dirname(output), { recursive: true });
		if (trace) await mkdir(dirname(trace), { recursive: true });

		let turns = 0;
		const events: unknown[] = [];
		const evidence: EvidenceState = { bodyReadFiles: new Set(), contentSearchCommands: [] };
		const repairReasons: string[] = [];
		const model = createModel(args.modelProvider, args.model, args.openaiBaseUrl);
		const streamFn =
			args.modelProvider === "mock" ? await createMockStreamFn(args.mockScript!) : createBridgeStreamFn(args);
		const agent = new Agent({
			streamFn,
			initialState: {
				systemPrompt: DEFAULT_CONTRACT.systemPrompt,
				model,
				thinkingLevel: "off",
				tools: [createReadTool(sandbox, evidence), createBashTool(sandbox, scratch, evidence)],
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
			failure_reason: acceptedParsed
				? null
				: (finalIssue ?? "max_agent_steps reached before an acceptable final answer"),
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
			inspect_sandbox: "local",
			tool_sandbox: "bubblewrap",
			contract: {
				top_k: DEFAULT_CONTRACT.topK,
				max_agent_steps: args.maxAgentSteps,
				tool_allowlist: ["read", "bash"],
				tool_sandbox: "bubblewrap",
				max_scratch_bytes: DEFAULT_CONTRACT.maxScratchBytes,
			},
		};
		await writeFile(output, `${JSON.stringify(result, null, 2)}\n`, "utf8");
		if (trace) {
			await writeFile(
				trace,
				`${JSON.stringify(
					{
						argv: redactArgv(process.argv.slice(2)),
						cwd: process.cwd(),
						workspace: sandbox.workspace,
						output,
						inspect_sandbox: "local",
						tool_sandbox: "bubblewrap",
						events,
						messages: agent.state.messages,
					},
					null,
					2,
				)}\n`,
				"utf8",
			);
		}
		if (!acceptedParsed) {
			console.error(`Invalid DCI final answer from ${basename(fileURLToPath(import.meta.url))}: ${finalText}`);
			process.exitCode = 2;
		}
	} finally {
		await rm(scratch, { recursive: true, force: true });
	}
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});
