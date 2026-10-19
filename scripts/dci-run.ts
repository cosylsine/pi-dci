#!/usr/bin/env node
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { constants as fsConstants, readFileSync, renameSync, writeFileSync } from "node:fs";
import { access, lstat, mkdir, mkdtemp, readFile, readdir, realpath, rm, stat } from "node:fs/promises";
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
} from "@earendil-works/pi-ai";
import { Unsafe } from "typebox";
import { Value } from "typebox/value";
import type { TSchema } from "typebox";

type CanonicalToolDefinition = {
	name: string;
	description: string;
	input_schema: TSchema;
	result_max_bytes: number;
};

type BudgetControlPolicy = {
	version: string;
	reserve_tokens: number;
	forecast_bytes_per_token?: number;
	forecast_request_safety_tokens?: number;
	repeat_threshold: number;
	max_search_turns_without_read: number;
	max_required_read_turns: number;
	initial_prompt: string;
	loop_prompt: string;
	read_prompt: string;
	resume_prompt: string;
	final_prompt: string;
	final_repair_prompt: string;
};

type CanonicalContract = {
	evaluation_track?: "document" | "code";
	benchmark_contract_version: string;
	protocol_profile?: string;
	budget_control?: BudgetControlPolicy;
	search_policy?: { version: string; ripgrep_config: string; prompt: string };
	pi_final_result_version: string;
	trajectory_schema_version: string;
	top_k: number;
	max_agent_steps: number;
	max_final_repair_turns: number;
	sample_timeout_seconds: number;
	tool_result_max_bytes: number;
	max_scratch_bytes: number;
	max_total_model_tokens: number;
	max_final_output_tokens: number;
	max_model_output_tokens: number;
	generation_config: { temperature: number; max_tokens: number; seed: number; enable_thinking: boolean };
	max_corpus_document_bytes: number;
	tool_allowlist: string[];
	tool_definitions: CanonicalToolDefinition[];
	failure_kinds: string[];
	final_answer_schema: Record<string, unknown>;
	system_prompt: string;
	query_prompt_template: string;
	task_instruction_template: string;
	repair_prompt_template: string;
};

const CONTRACT_PATH = resolve(dirname(fileURLToPath(import.meta.url)), "../../dci_bench/protocol/contract.json");
const EVALUATION_TRACK = process.env.DCI_EVALUATION_TRACK ?? "document";
if (EVALUATION_TRACK !== "document" && EVALUATION_TRACK !== "code") throw new Error("Unknown DCI_EVALUATION_TRACK");
const codeContractPath = process.env.DCI_CODE_RUNTIME_CONTRACT;
if (EVALUATION_TRACK === "code" && !codeContractPath) throw new Error("Code evaluation requires DCI_CODE_RUNTIME_CONTRACT");
const DEFAULT_CONTRACT = JSON.parse(readFileSync(EVALUATION_TRACK === "code" ? codeContractPath! : CONTRACT_PATH, "utf8")) as CanonicalContract;
if (EVALUATION_TRACK === "code" && (DEFAULT_CONTRACT.evaluation_track !== "code" || DEFAULT_CONTRACT.budget_control)) {
	throw new Error("Code runtime contract must declare evaluation_track=code and must not copy document budget_control");
}
const PROTOCOL_PROFILE = EVALUATION_TRACK === "code" ? "code-v1" : process.env.DCI_PROTOCOL_PROFILE ?? "v3";
if (EVALUATION_TRACK === "document" && PROTOCOL_PROFILE !== "v3") {
	const profiles = JSON.parse(readFileSync(join(dirname(CONTRACT_PATH), "pilot_profiles.json"), "utf8")) as Record<
		string,
		{ benchmark_contract_version?: string; generation_config?: Partial<CanonicalContract["generation_config"]>; budget_control?: string;
			budget_control_overrides?: Partial<BudgetControlPolicy>; policy?: BudgetControlPolicy;
			search_policy?: string | CanonicalContract["search_policy"] }
	>;
	const profile = profiles[PROTOCOL_PROFILE];
	if (!profile || PROTOCOL_PROFILE.startsWith("_")) throw new Error(`Unknown DCI_PROTOCOL_PROFILE: ${PROTOCOL_PROFILE}`);
	DEFAULT_CONTRACT.benchmark_contract_version = profile.benchmark_contract_version ?? "dci-mvp-v3.3-pilot";
	DEFAULT_CONTRACT.protocol_profile = PROTOCOL_PROFILE;
	Object.assign(DEFAULT_CONTRACT.generation_config, profile.generation_config ?? {});
	if (profile.budget_control) {
		const policy = profiles[profile.budget_control].policy;
		if (!policy) throw new Error(`Missing budget policy: ${profile.budget_control}`);
		DEFAULT_CONTRACT.budget_control = { ...policy, ...profile.budget_control_overrides };
	}
	if (typeof profile.search_policy === "string") {
		const policy = profiles[profile.search_policy].search_policy;
		if (!policy || typeof policy === "string") throw new Error(`Missing search policy: ${profile.search_policy}`);
		DEFAULT_CONTRACT.search_policy = { ...policy };
		if (readFileSync(join(dirname(CONTRACT_PATH), "stable-rg.conf"), "utf8") !== policy.ripgrep_config) {
			throw new Error("stable-rg.conf does not match the protocol search policy");
		}
	}
}
const CONTRACT_SHA256 = createHash("sha256").update(JSON.stringify(sortJsonKeys(DEFAULT_CONTRACT))).digest("hex");
if (process.env.DCI_PROTOCOL_CONTRACT_SHA256 && process.env.DCI_PROTOCOL_CONTRACT_SHA256 !== CONTRACT_SHA256) {
	throw new Error("Python/Pi protocol contract hash mismatch");
}
const PROCESS_STARTED_AT = Date.now();

type Args = {
	codeFiles?: string;
	preflightOnly: boolean;
	queryFile: string;
	taskInstructionFile?: string;
	workspace: string;
	output: string;
	trace?: string;
	modelProvider: "bridge" | "mock";
	mockScript?: string;
	openaiBaseUrl: string;
	openaiApiKey: string;
	model: string;
	maxAgentSteps: number;
	contextWindow: number;
	timeoutSeconds: number;
	deadlineAtMs?: number;
};

type MockStep =
	| { type: "tool_call"; name: string; arguments: Record<string, unknown>; id?: string; usage?: MockUsage }
	| { type: "final"; text: string; usage?: MockUsage };

type MockUsage = {
	input?: number;
	output?: number;
	cacheRead?: number;
	cacheWrite?: number;
	cache_read?: number;
	cache_write?: number;
	input_tokens?: number;
	output_tokens?: number;
	prompt_tokens?: number;
	completion_tokens?: number;
};

type UsageSummary = {
	available: boolean;
	input_tokens: number | null;
	output_tokens: number | null;
	cache_read_tokens: number | null;
	cache_write_tokens: number | null;
	total_tokens: number | null;
	total_model_tokens: number | null;
	turns: number;
};

type FailureKind = CanonicalContract["failure_kinds"][number];

type StructuredFailure = { kind: Exclude<FailureKind, "none">; message: string } | null;

function sortJsonKeys(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(sortJsonKeys);
	if (value !== null && typeof value === "object") {
		const record = value as Record<string, unknown>;
		return Object.fromEntries(Object.keys(record).sort().map((key) => [key, sortJsonKeys(record[key])]));
	}
	return value;
}

function prettyJson(value: unknown): string {
	return `${JSON.stringify(sortJsonKeys(value), null, 2)}\n`;
}

function makeFailure(kind: Exclude<FailureKind, "none">, message: string): StructuredFailure {
	if (!DEFAULT_CONTRACT.failure_kinds.includes(kind)) {
		throw new Error(`Canonical contract does not allow failure kind ${kind}`);
	}
	return { kind, message };
}

function usage(input = 0, output = 0, cacheRead = 0, cacheWrite = 0) {
	return {
		input,
		output,
		cacheRead,
		cacheWrite,
		totalTokens: input + output,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function numberOrUndefined(value: unknown): number | undefined {
	return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function parseUsage(raw: unknown): { available: boolean; value: ReturnType<typeof usage> } {
	if (!raw || typeof raw !== "object") return { available: false, value: usage() };
	const payload = raw as Record<string, unknown>;
	const input = numberOrUndefined(payload.input ?? payload.prompt_tokens ?? payload.input_tokens);
	const output = numberOrUndefined(payload.output ?? payload.completion_tokens ?? payload.output_tokens);
	const promptDetails = payload.prompt_tokens_details;
	const promptDetailsCacheRead =
		promptDetails && typeof promptDetails === "object"
			? (promptDetails as Record<string, unknown>).cached_tokens
			: undefined;
	const cacheRead =
		numberOrUndefined(
			payload.cacheRead ??
				payload.cache_read_tokens ??
				payload.cache_read ??
				payload.cache_read_input_tokens ??
				promptDetailsCacheRead,
		) ?? 0;
	const cacheWrite =
		numberOrUndefined(
			payload.cacheWrite ?? payload.cache_write_tokens ?? payload.cache_write ?? payload.cache_creation_input_tokens,
		) ?? 0;
	if (input === undefined || output === undefined) return { available: false, value: usage() };
	return { available: true, value: usage(input, output, cacheRead, cacheWrite) };
}

function unavailableUsage(): ReturnType<typeof usage> & { usageAvailable: boolean } {
	return { ...usage(), usageAvailable: false };
}

function availableUsage(value: ReturnType<typeof usage>): ReturnType<typeof usage> & { usageAvailable: boolean } {
	return { ...value, usageAvailable: true };
}

function readMessageUsage(message: AssistantMessage): { available: boolean; value: ReturnType<typeof usage> } {
	const messageWithMarker = message as AssistantMessage & { usageAvailable?: unknown };
	const usageWithMarker = message.usage as AssistantMessage["usage"] & { usageAvailable?: unknown };
	if (
		messageWithMarker.usageAvailable === false ||
		usageWithMarker.usageAvailable === false ||
		message.stopReason === "error" ||
		message.stopReason === "aborted"
	) {
		return { available: false, value: usage() };
	}
	const parsed = parseUsage(message.usage);
	return { available: parsed.available, value: parsed.value };
}

function usageSummary(messages: readonly Message[], totalModelTokens: number): UsageSummary {
	const assistantMessages = messages.filter((message): message is AssistantMessage => message.role === "assistant");
	const usages = assistantMessages.map(readMessageUsage);
	const available = usages.every((item) => item.available);
	const input = available ? usages.reduce((sum, item) => sum + item.value.input, 0) : null;
	const output = available ? usages.reduce((sum, item) => sum + item.value.output, 0) : null;
	const cacheRead = available ? usages.reduce((sum, item) => sum + item.value.cacheRead, 0) : null;
	const cacheWrite = available ? usages.reduce((sum, item) => sum + item.value.cacheWrite, 0) : null;
	const total = available ? usages.reduce((sum, item) => sum + item.value.totalTokens, 0) : null;
	return {
		available,
		input_tokens: input,
		output_tokens: output,
		cache_read_tokens: cacheRead,
		cache_write_tokens: cacheWrite,
		total_tokens: total,
		total_model_tokens: available ? totalModelTokens : null,
		turns: assistantMessages.length,
	};
}

function usageByTurn(messages: readonly Message[]): Array<Record<string, number | boolean | null>> {
	let turn = 0;
	return messages.flatMap((message) => {
		if (message.role !== "assistant") return [];
		turn++;
		const current = readMessageUsage(message);
		return [
			{
				turn,
				available: current.available,
				input_tokens: current.available ? current.value.input : null,
				output_tokens: current.available ? current.value.output : null,
				total_tokens: current.available ? current.value.totalTokens : null,
			},
		];
	});
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
	if (EVALUATION_TRACK !== "code" && values.has("code-files")) throw new Error("--code-files requires code evaluation");
	const timeoutValue = values.get("timeout-seconds") ?? String(DEFAULT_CONTRACT.sample_timeout_seconds);
	const timeoutSeconds = typeof timeoutValue === "string" ? Number(timeoutValue) : NaN;
	if (!Number.isFinite(timeoutSeconds) || timeoutSeconds <= 0) throw new Error("--timeout-seconds must be positive");
	const deadlineValue = values.get("deadline-at-ms");
	const deadlineAtMs = deadlineValue === undefined ? undefined : typeof deadlineValue === "string" ? Number(deadlineValue) : NaN;
	if (deadlineAtMs !== undefined && (!Number.isFinite(deadlineAtMs) || deadlineAtMs <= 0)) {
		throw new Error("--deadline-at-ms must be a positive epoch timestamp");
	}
	return {
		codeFiles: EVALUATION_TRACK === "code" ? required("code-files") : undefined,
		preflightOnly,
		queryFile: preflightOnly ? "" : required("query-file"),
		taskInstructionFile:
			typeof values.get("task-instruction-file") === "string" ? String(values.get("task-instruction-file")) : undefined,
		workspace: required("workspace"),
		output: preflightOnly ? "" : required("output"),
		trace: typeof values.get("trace") === "string" ? String(values.get("trace")) : undefined,
		modelProvider: values.get("mock-script") ? "mock" : "bridge",
		mockScript: typeof values.get("mock-script") === "string" ? String(values.get("mock-script")) : undefined,
		openaiBaseUrl: String(values.get("openai-base-url") ?? process.env.OPENAI_BASE_URL ?? "http://localhost:13131/v1"),
		openaiApiKey: String(values.get("openai-api-key") ?? process.env.OPENAI_API_KEY ?? "inspect"),
		model: String(values.get("model") ?? "inspect"),
		maxAgentSteps: positiveInt("max-agent-steps", DEFAULT_CONTRACT.max_agent_steps),
		contextWindow: positiveInt("context-window", DEFAULT_CONTRACT.max_total_model_tokens),
		timeoutSeconds,
		deadlineAtMs,
	};
}

function toolDefinition(name: string): CanonicalToolDefinition {
	const definition = DEFAULT_CONTRACT.tool_definitions.find((item) => item.name === name);
	if (!definition) throw new Error(`Canonical contract is missing the ${name} tool definition`);
	return definition;
}

type ReadToolArguments = { path: string; offset?: number; limit?: number };
type BashToolArguments = { command: string; timeout_seconds?: number };

const readDefinition = toolDefinition("read");
const bashDefinition = toolDefinition("bash");
const readSchema = Unsafe<ReadToolArguments>(readDefinition.input_schema);
const bashSchema = Unsafe<BashToolArguments>(bashDefinition.input_schema);

export function utf8Prefix(text: string, maxBytes: number): string {
	const bytes = Buffer.from(text, "utf8");
	if (bytes.length <= maxBytes) return text;
	let end = Math.max(0, maxBytes);
	while (end > 0 && (bytes[end] & 0xc0) === 0x80) end--;
	return bytes.subarray(0, end).toString("utf8");
}

export function truncate(text: string, maxBytes = DEFAULT_CONTRACT.tool_result_max_bytes): string {
	if (Buffer.byteLength(text, "utf8") <= maxBytes) return text;
	const marker = "\n[truncated]";
	return utf8Prefix(text, Math.max(0, maxBytes - Buffer.byteLength(marker))) + marker;
}

function writeJsonAtomicSync(path: string, payload: unknown): void {
	const temporary = `${path}.${process.pid}.tmp`;
	writeFileSync(temporary, prettyJson(payload), "utf8");
	renameSync(temporary, path);
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
			if (total > DEFAULT_CONTRACT.max_scratch_bytes) return total;
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
	rgConfig?: string;
	codeFiles?: CodeFiles;
};

type CodeFileMetadata = { sha256: string; line_count: number; content_tokens?: number };
type CodeFiles = Record<string, CodeFileMetadata>;
type CodeSpan = { path: string; start_line: number; end_line: number };
type CodeOutput = { ranked_files: string[]; abstain: boolean; spans?: CodeSpan[] };

function exactCodePath(value: unknown): value is string {
	return typeof value === "string" && value.length > 0 && !/[\\\u0000-\u001f\u007f]/.test(value) &&
		!value.split("/").some((part) => part === "" || part === "." || part === "..") && !/^[A-Za-z]:/.test(value);
}

function sourceLines(text: string): string[] {
	if (!text) return [];
	const lines = text.split("\n");
	if (text.endsWith("\n")) lines.pop();
	return lines;
}

export function parseCodeFiles(value: unknown): CodeFiles {
	if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).length === 0) {
		throw new Error("Code file manifest must be a nonempty object");
	}
	const files = Object.create(null) as CodeFiles;
	for (const [path, raw] of Object.entries(value)) {
		if (!exactCodePath(path) || !raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error(`Invalid code file: ${path}`);
		const metadata = raw as Record<string, unknown>;
		if (Object.keys(metadata).some((key) => !["sha256", "line_count", "content_tokens"].includes(key)) ||
			typeof metadata.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(metadata.sha256) ||
			typeof metadata.line_count !== "number" || !Number.isSafeInteger(metadata.line_count) || metadata.line_count < 0 ||
			(metadata.content_tokens !== undefined && (typeof metadata.content_tokens !== "number" ||
				!Number.isSafeInteger(metadata.content_tokens) || metadata.content_tokens < 0))) {
			throw new Error(`Invalid code file metadata: ${path}`);
		}
		files[path] = metadata as CodeFileMetadata;
	}
	return files;
}

export async function validateCodeCorpus(corpus: string, files: CodeFiles): Promise<void> {
	const pending = [""];
	const seen = new Set<string>();
	while (pending.length) {
		const relative = pending.pop()!;
		for (const entry of await readdir(join(corpus, relative), { withFileTypes: true })) {
			const path = relative ? `${relative}/${entry.name}` : entry.name;
			const info = await lstat(join(corpus, path));
			if (info.isSymbolicLink()) throw new Error(`Code corpus rejects symlinks: ${path}`);
			if (info.isDirectory()) {
				if (!Object.keys(files).some((file) => file.startsWith(`${path}/`))) throw new Error(`Unexpected code corpus directory: ${path}`);
				pending.push(path);
				continue;
			}
			if (!info.isFile() || info.nlink !== 1) throw new Error(`Code corpus requires regular files without hard links: ${path}`);
			if (!Object.hasOwn(files, path)) throw new Error(`Unexpected code corpus file: ${path}`);
			if (info.size > DEFAULT_CONTRACT.max_corpus_document_bytes) throw new Error(`Code corpus file exceeds the size limit: ${path}`);
			const bytes = await readFile(join(corpus, path));
			const text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
			if (createHash("sha256").update(bytes).digest("hex") !== files[path].sha256 || sourceLines(text).length !== files[path].line_count) {
				throw new Error(`Code corpus hash or LF line count mismatch: ${path}`);
			}
			seen.add(path);
		}
	}
	for (const path of Object.keys(files)) if (!seen.has(path)) throw new Error(`Missing code corpus file: ${path}`);
}

async function validateCorpusSandbox(workspace: string, codeFiles?: CodeFiles): Promise<CorpusSandbox> {
	const resolvedWorkspace = await realpath(workspace);
	const corpus = resolve(resolvedWorkspace, "corpus");
	const corpusInfo = await lstat(corpus);
	if (corpusInfo.isSymbolicLink() || !corpusInfo.isDirectory()) {
		throw new Error(`Corpus must be a real directory: ${corpus}`);
	}
	if (codeFiles) await validateCodeCorpus(corpus, codeFiles);
	else for (const entry of await readdir(corpus, { withFileTypes: true })) {
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
		if (entryInfo.size > DEFAULT_CONTRACT.max_corpus_document_bytes) {
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
	const rgCandidate = DEFAULT_CONTRACT.search_policy ? "/usr/bin/rg" : await findExecutable("rg");
	let rg: string | undefined;
	if (rgCandidate) {
		try {
			rg = await validateTrustedExecutable(rgCandidate, "ripgrep");
		} catch {
			// grep remains available from the read-only system mount.
		}
	}
	if (DEFAULT_CONTRACT.search_policy && !rg) throw new Error("stable-rg requires a trusted ripgrep executable");
	return { workspace: resolvedWorkspace, corpus: await realpath(corpus), bwrap, rg, codeFiles,
		rgConfig: DEFAULT_CONTRACT.search_policy ? join(dirname(CONTRACT_PATH), "stable-rg.conf") : undefined };
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
	if (sandbox.rgConfig) args.push("--ro-bind", sandbox.rgConfig, "/sandbox-bin/rg.conf",
		"--setenv", "RIPGREP_CONFIG_PATH", "/sandbox-bin/rg.conf");
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
			"test -r /workspace/corpus && test ! -e /mnt && test ! -e /root && test ! -e /proc && test ! -e /tmp && test -w /scratch && test -w \"$TMPDIR\"" +
			(sandbox.rgConfig ? " && test -r \"$RIPGREP_CONFIG_PATH\" && test ! -w \"$RIPGREP_CONFIG_PATH\" && rg --files /workspace/corpus >/dev/null" : "");
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
	readReceipts: Array<{ path: string; doc_id?: string; offset_codepoints?: number; returned_codepoints?: number;
		start_line?: number; end_line?: number; last_line_complete?: boolean; returned_bytes: number; content_sha256: string }>;
};

// Recover IDs only from trusted workspace filenames, never from submitted IDs.
// This preserves original IDs such as report.md, a%2Fb and 100% exactly.
export function corpusDocumentIds(filenames: string[]): { ids: Set<string>; aliases: Map<string, string> } {
	const ids = new Set<string>();
	const aliases = new Map<string, string>();
	for (const filename of filenames) {
		if (!filename.endsWith(".md")) throw new Error(`Not a corpus filename: ${filename}`);
		const stem = filename.slice(0, -3);
		const id = decodeURIComponent(stem);
		if (!id || ids.has(id)) throw new Error(`Empty or duplicate original corpus ID: ${filename}`);
		ids.add(id);
		aliases.set(filename, id);
		aliases.set(stem, id);
	}
	return { ids, aliases };
}

export function documentIdIssue(parsed: ParsedFinal, corpusIds: ReturnType<typeof corpusDocumentIds>): string | undefined {
	const unknown = parsed.ranked_doc_ids.filter((id) => !corpusIds.ids.has(id));
	if (!unknown.length) return undefined;
	const hints = unknown.flatMap((id) => {
		const original = corpusIds.aliases.get(id);
		return original === undefined ? [] : [{ submitted: id, doc_id: original }];
	});
	return truncate(`Unknown original document IDs: ${JSON.stringify(unknown)}. ` +
		`Exact filename-to-doc_id mappings: ${JSON.stringify(hints)}. Copy original doc_id values; do not submit filenames.`, 4000);
}

function workspaceRelativePath(workspace: string, path: string): string {
	const workspaceRoot = resolve(workspace);
	const full = resolve(path);
	return full === workspaceRoot || !full.startsWith(`${workspaceRoot}/`) ? full : full.slice(workspaceRoot.length + 1);
}

function hasBodyEvidence(evidence: EvidenceState): boolean {
	return evidence.readReceipts.length > 0;
}

function stripModelNoise(text: string): string {
	return text
		.trim()
		.replace(/^(?:(?:<think>[\s\S]*?<\/think>|<<content-internal>[\s\S]*?<\/content-internal>>)\s*)+/, "")
		.trim();
}

function parseAdaptedJson(text: string): unknown {
	try {
		return JSON.parse(text);
	} catch {
		// Only adapt text outside JSON. Literal thinking tags inside an ID or
		// shell argument belong to the model's payload and must remain intact.
		return JSON.parse(stripModelNoise(text));
	}
}

function parsePseudoToolCall(text: string): ToolCall | undefined {
	try {
		const value = parseAdaptedJson(text);
		if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
		const parsed = value as Record<string, unknown>;
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
	// Search commands commonly cd into corpus and return bare filenames. Treat
	// only a single literal .md filename as corpus-relative; never decode it or
	// rebase paths containing directories/traversal into the corpus.
	const filename = requested.startsWith("./") ? requested.slice(2) : requested;
	const bareFilename = filename.endsWith(".md") && basename(filename) === filename &&
		!filename.includes("\\") && !filename.includes("\0");
	const mapped = requested === "/workspace/corpus" || requested.startsWith("/workspace/corpus/")
		? `${sandbox.corpus}${requested.slice("/workspace/corpus".length)}`
		: bareFilename ? join(sandbox.corpus, filename) : requested;
	const full = mapped.startsWith("/") ? resolve(mapped) : resolve(sandbox.workspace, mapped);
	if (full !== sandbox.corpus && !full.startsWith(`${sandbox.corpus}${sep}`)) {
		throw new Error(`Path escapes corpus: ${requested}. Use an exact corpus filename or /workspace/corpus/<filename>.md; shell cd does not change read paths.`);
	}
	const info = await lstat(full).catch((error: unknown) => {
		if (error instanceof Error && "code" in error && error.code === "ENOENT") {
			throw new Error(`Corpus path not found: ${requested}. Copy the exact filename from search results, including its .md suffix and percent escapes; /workspace/corpus/<filename> is also accepted.`);
		}
		throw error;
	});
	if (info.isSymbolicLink()) throw new Error(`Symlinks are not allowed in corpus reads: ${requested}`);
	if (!info.isDirectory() && (!info.isFile() || info.nlink !== 1)) {
		throw new Error(`Corpus reads require a regular file without hard links: ${requested}`);
	}
	const canonical = await realpath(full);
	if (canonical !== sandbox.corpus && !canonical.startsWith(`${sandbox.corpus}${sep}`)) {
		throw new Error(`Path escapes corpus after symlink resolution: ${requested}`);
	}
	return canonical;
}

export function createReadTool(sandbox: CorpusSandbox, evidence: EvidenceState): AgentTool<typeof readSchema> {
	return {
		name: "read",
		label: "read",
		description: readDefinition.description,
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
			if (info.size > DEFAULT_CONTRACT.max_corpus_document_bytes) {
				throw new Error(`Corpus document exceeds the size limit: ${params.path}`);
			}
			const raw = await readFile(path, "utf8");
			const offset = Math.max(0, Math.floor(params.offset ?? 0));
			const limit = Math.max(1, Math.floor(params.limit ?? DEFAULT_CONTRACT.tool_result_max_bytes));
			const selected = Array.from(raw).slice(offset, offset + limit).join("");
			const docId = decodeURIComponent(basename(path).slice(0, -3));
			const header = `${JSON.stringify({ doc_id: docId })}\n`;
			const bodyBytes = DEFAULT_CONTRACT.tool_result_max_bytes - Buffer.byteLength(header);
			const text = header + truncate(selected, bodyBytes);
			const visibleContent = Buffer.byteLength(selected) > bodyBytes
				? utf8Prefix(selected, bodyBytes - Buffer.byteLength("\n[truncated]"))
				: selected;
			const relativePath = workspaceRelativePath(sandbox.workspace, path);
			const bodyEvidence = visibleContent.trim().length > 0;
			const receipt = bodyEvidence ? {
				path: `/workspace/${relativePath}`,
				doc_id: docId,
				offset_codepoints: offset,
				returned_codepoints: Array.from(visibleContent).length,
				returned_bytes: Buffer.byteLength(visibleContent),
				content_sha256: createHash("sha256").update(visibleContent).digest("hex"),
			} : undefined;
			if (receipt) {
				evidence.bodyReadFiles.add(relativePath);
				evidence.readReceipts.push(receipt);
			}
			return {
				content: [{ type: "text", text }],
				details: { path: `/workspace/${relativePath}`, doc_id: docId, bytes: Buffer.byteLength(text), bodyEvidence, receipt },
			};
		},
	};
}

type CodeReadArguments = { path: string; start_line?: number; end_line?: number };
const codeReadSchema = Unsafe<CodeReadArguments>(EVALUATION_TRACK === "code" ? readDefinition.input_schema : {
	type: "object", additionalProperties: false, required: ["path"], properties: {
		path: { type: "string", minLength: 1 }, start_line: { type: "integer", minimum: 1 },
		end_line: { type: "integer", minimum: 1 },
	},
});

export function createCodeReadTool(sandbox: CorpusSandbox, evidence: EvidenceState): AgentTool<typeof codeReadSchema> {
	return {
		name: "read", label: "read", description: EVALUATION_TRACK === "code" ? readDefinition.description :
			"Read exact repository paths using 1-based inclusive source line ranges.",
		parameters: codeReadSchema, executionMode: "sequential",
		prepareArguments: (args) => Value.Parse(codeReadSchema, args),
		execute: async (_toolCallId, params) => {
			const files = sandbox.codeFiles;
			if (!files) throw new Error("Code read requires a validated file manifest");
			const requested = params.path;
			const path = Object.hasOwn(files, requested) ? requested : requested.startsWith("/workspace/corpus/")
				? requested.slice("/workspace/corpus/".length) : requested.startsWith("corpus/") ? requested.slice("corpus/".length) : requested;
			if (!exactCodePath(path) || !Object.hasOwn(files, path)) throw new Error(`Unknown exact repository path: ${requested}`);
			let full = sandbox.corpus;
			for (const part of path.split("/")) {
				full = join(full, part);
				if ((await lstat(full)).isSymbolicLink()) throw new Error(`Code read rejects symlinks: ${path}`);
			}
			const info = await lstat(full);
			if (!info.isFile() || info.nlink !== 1 || info.size > DEFAULT_CONTRACT.max_corpus_document_bytes) {
				throw new Error(`Code read requires a bounded regular file without hard links: ${path}`);
			}
			const bytes = await readFile(full);
			if (createHash("sha256").update(bytes).digest("hex") !== files[path].sha256) throw new Error(`Code file changed: ${path}`);
			const raw = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
			const lines = sourceLines(raw);
			const start = params.start_line ?? 1;
			const end = params.end_line ?? lines.length;
			if (!Number.isSafeInteger(start) || start < 1 || !Number.isSafeInteger(end) ||
				(end < start && !(lines.length === 0 && start === 1 && params.end_line === undefined)) || end > lines.length || start > Math.max(1, lines.length)) {
				throw new Error(`Invalid source line range for ${path}`);
			}
			const selected = lines.slice(start - 1, end).join("\n");
			// Reserve metadata space before truncating source bytes. A partial last
			// line is explicit and must never be mistaken for a fully read line.
			const headerReserve = Buffer.byteLength(JSON.stringify({ path, start_line: start, end_line: end, last_line_complete: false })) + 2;
			const bodyBudget = DEFAULT_CONTRACT.tool_result_max_bytes - headerReserve;
			if (bodyBudget <= Buffer.byteLength("\n[truncated]")) throw new Error("Code path metadata exceeds the output byte limit");
			const clipped = Buffer.byteLength(selected) > bodyBudget;
			const visible = clipped ? utf8Prefix(selected, bodyBudget - Buffer.byteLength("\n[truncated]")) : selected;
			const returnedEnd = !clipped ? end : visible.length ? start + sourceLines(visible).length - 1 : start - 1;
			const complete = !clipped || visible.endsWith("\n");
			const header = JSON.stringify({ path, start_line: start, end_line: returnedEnd, last_line_complete: complete });
			const text = `${header}\n${visible}${clipped ? "\n[truncated]" : ""}`;
			const bodyEvidence = visible.trim().length > 0;
			const receipt = bodyEvidence ? { path, start_line: start, end_line: returnedEnd, last_line_complete: complete,
				returned_bytes: Buffer.byteLength(visible), content_sha256: createHash("sha256").update(visible).digest("hex") } : undefined;
			if (receipt) { evidence.bodyReadFiles.add(`corpus/${path}`); evidence.readReceipts.push(receipt); }
			return { content: [{ type: "text", text }], details: { path, bytes: Buffer.byteLength(text), bodyEvidence, receipt } };
		},
	};
}

export function createBashTool(sandbox: CorpusSandbox, scratch: string, evidence: EvidenceState): AgentTool<typeof bashSchema> {
	return {
		name: "bash",
		label: "bash",
		description: bashDefinition.description,
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
					// An unwritten stdin pipe makes pathless rg wait for input instead of searching cwd.
					stdio: ["ignore", "pipe", "pipe"],
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
					const remaining = DEFAULT_CONTRACT.tool_result_max_bytes - capturedBytes;
					if (remaining > 0) chunks.push(chunk.subarray(0, remaining));
					capturedBytes += chunk.length;
					if (capturedBytes > DEFAULT_CONTRACT.tool_result_max_bytes && !outputLimitExceeded) {
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
					if ((await directoryUsageBytes(scratch)) > DEFAULT_CONTRACT.max_scratch_bytes) {
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
				const stdout = new TextDecoder("utf-8", { fatal: false }).decode(Buffer.concat(stdoutChunks), { stream: true });
				const stderr = new TextDecoder("utf-8", { fatal: false }).decode(Buffer.concat(stderrChunks), { stream: true });
				const terminationReason = outputLimitExceeded
					? "[output limit exceeded; process terminated]"
					: scratchLimitExceeded
						? "[scratch limit exceeded; process terminated]"
					: controller.signal.aborted
						? "[timeout exceeded; process terminated]"
						: signal?.aborted
							? "[aborted; process terminated]"
							: "";
				const commandOutput = truncate([stdout, stderr ? `STDERR:\n${stderr}` : ""].filter(Boolean).join("\n"), DEFAULT_CONTRACT.tool_result_max_bytes - 256);
				const text = truncate([commandOutput, terminationReason, `exit_code=${exitCode}`].filter(Boolean).join("\n"));
				const bodyEvidence = false; // Arbitrary shell text is never proof of a corpus read.
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

function parseRankedIds(value: unknown): string[] | undefined {
	if (
		Array.isArray(value) &&
		value.length > 0 &&
		value.length <= DEFAULT_CONTRACT.top_k &&
		value.every((item: unknown) => typeof item === "string" && item.length > 0)
	) {
		const ranked = value as string[];
		if (new Set(ranked).size === ranked.length) return ranked;
	}
	return undefined;
}

export function parseFinalJson(text: string): ParsedFinal | undefined {
	try {
		const value = parseAdaptedJson(text);
		if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
		const parsed = value as Record<string, unknown>;
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

function finalAcceptanceIssue(parsed: ParsedFinal | undefined, evidence: EvidenceState, corpusIds: ReturnType<typeof corpusDocumentIds>): string | undefined {
	if (!parsed) return "final answer did not match DCI ranked_doc_ids JSON schema";
	if (!hasBodyEvidence(evidence)) {
		return "final answer requires a nonempty corpus read tool receipt";
	}
	return documentIdIssue(parsed, corpusIds);
}

export function parseCodeFinalJson(text: string, files: CodeFiles, maxFiles = DEFAULT_CONTRACT.top_k): CodeOutput | undefined {
	try {
		let source = text;
		try { JSON.parse(source); } catch { source = stripModelNoise(source); }
		const value: unknown = JSON.parse(source);
		// JSON.parse silently accepts duplicate object keys. The host scorer is
		// strict, so reject ambiguous payloads here before reporting success.
		const tokens = source.match(/"(?:\\.|[^"\\])*"|[{}\[\]:,]/g) ?? [];
		const objects: Array<Set<string> | null> = [];
		for (let index = 0; index < tokens.length; index++) {
			const token = tokens[index];
			if (token === "{") objects.push(new Set());
			else if (token === "[") objects.push(null);
			else if (token === "}" || token === "]") objects.pop();
			else if (token.startsWith('"') && tokens[index + 1] === ":") {
				const keys = objects[objects.length - 1];
				const key = JSON.parse(token) as string;
				if (keys?.has(key)) return undefined;
				keys?.add(key);
			}
		}
		if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
		const parsed = value as Record<string, unknown>;
		if (Object.keys(parsed).some((key) => !["ranked_files", "abstain", "spans"].includes(key)) ||
			!Array.isArray(parsed.ranked_files) || parsed.ranked_files.length > maxFiles ||
			!parsed.ranked_files.every((path: unknown) => exactCodePath(path) && Object.hasOwn(files, path)) ||
			new Set(parsed.ranked_files).size !== parsed.ranked_files.length || typeof parsed.abstain !== "boolean" ||
			parsed.abstain !== (parsed.ranked_files.length === 0)) return undefined;
		if (parsed.spans !== undefined) {
			if (!Array.isArray(parsed.spans)) return undefined;
			for (const span of parsed.spans) {
				if (!span || typeof span !== "object" || Array.isArray(span)) return undefined;
				const row = span as Record<string, unknown>;
				if (Object.keys(row).length !== 3 || !exactCodePath(row.path) || !parsed.ranked_files.includes(row.path) ||
					typeof row.start_line !== "number" || !Number.isSafeInteger(row.start_line) || row.start_line < 1 ||
					typeof row.end_line !== "number" || !Number.isSafeInteger(row.end_line) || row.end_line < row.start_line ||
					row.end_line > files[row.path].line_count) return undefined;
			}
		}
		return parsed as CodeOutput;
	} catch { return undefined; }
}

type RuntimeFinal = ParsedFinal | CodeOutput;

function parseRuntimeFinal(text: string, sandbox: CorpusSandbox): RuntimeFinal | undefined {
	return sandbox.codeFiles ? parseCodeFinalJson(text, sandbox.codeFiles) : parseFinalJson(text);
}

function runtimeFinalIssue(parsed: RuntimeFinal | undefined, evidence: EvidenceState,
	corpusIds: ReturnType<typeof corpusDocumentIds> | undefined): string | undefined {
	if (!corpusIds) {
		if (!parsed || !("ranked_files" in parsed)) return "final answer must contain exact ranked_files, abstain, and optional valid source spans";
		return hasBodyEvidence(evidence) ? undefined : "final answer, including abstention, requires a nonempty source read tool receipt";
	}
	return finalAcceptanceIssue(parsed as ParsedFinal | undefined, evidence, corpusIds);
}

function createModel(provider: Args["modelProvider"], id: string, baseUrl: string, contextWindow: number): Model<Api> {
	return {
		id,
		name: id,
		api: provider === "mock" ? "mock" : "openai-completions",
		provider: provider === "mock" ? "mock" : "inspect-bridge",
		baseUrl,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens: DEFAULT_CONTRACT.max_model_output_tokens,
	};
}

const CONTEXT_SAFETY_TOKENS = 4096;

function estimateNextInputTokens(context: Context, conservative = false, bytesPerToken = 1): number {
	let lastUsageIndex = -1;
	let estimated = 0;
	for (let index = context.messages.length - 1; index >= 0; index--) {
		const message = context.messages[index];
		if (message.role !== "assistant") continue;
		const current = readMessageUsage(message);
		if (!current.available) continue;
		lastUsageIndex = index;
		estimated = current.value.input + current.value.output;
		break;
	}
	const trailingMessages = context.messages.slice(lastUsageIndex + 1);
	const trailingSize = trailingMessages.reduce((total, message) => total +
		(conservative ? Buffer.byteLength(textOf(message.content), "utf8") : textOf(message.content).length), 0);
	if (lastUsageIndex < 0) {
		estimated += Math.ceil((context.systemPrompt?.length ?? 0) / 4);
		estimated += Math.ceil(JSON.stringify(context.tools ?? []).length / 4);
	}
	return estimated + Math.ceil(trailingSize / (conservative ? bytesPerToken : 4));
}

export function estimateBudgetReserve(context: Context, policy: Pick<BudgetControlPolicy,
	"reserve_tokens" | "forecast_bytes_per_token" | "forecast_request_safety_tokens">) {
	// Defaults preserve the v5 control arm exactly. Calibration affects only
	// early finalization, never the conservative pre-dispatch or actual-usage guards.
	const bytesPerToken = policy.forecast_bytes_per_token ?? 1;
	const safetyTokens = policy.forecast_request_safety_tokens ?? CONTEXT_SAFETY_TOKENS;
	if (!Number.isFinite(bytesPerToken) || bytesPerToken <= 0 || !Number.isSafeInteger(safetyTokens) || safetyTokens < 0) {
		throw new Error("Invalid budget forecast parameters");
	}
	const nextInput = estimateNextInputTokens(context, true, bytesPerToken);
	const requestOverhead = DEFAULT_CONTRACT.max_model_output_tokens + safetyTokens;
	const nextRequest = nextInput + requestOverhead;
	const toolGrowth = Math.ceil(DEFAULT_CONTRACT.tool_result_max_bytes / bytesPerToken);
	// Pay for one search/read, final submission and one repair. The next tool
	// result is replayed in both later requests. Multiple large tool results and
	// further repairs may cost more; actual usage remains the hard authority.
	const afterSearchInput = nextInput + DEFAULT_CONTRACT.max_model_output_tokens + toolGrowth;
	const reserve = Math.max(policy.reserve_tokens,
		nextRequest + (afterSearchInput + requestOverhead) +
			(afterSearchInput + DEFAULT_CONTRACT.max_model_output_tokens + requestOverhead));
	return { nextInput, nextRequest, toolGrowth, safetyTokens, bytesPerToken, reserve };
}

function failureFromAgentError(message: string): StructuredFailure {
	const lowered = message.toLowerCase();
	if (
		(lowered.includes("context") && (lowered.includes("length") || lowered.includes("overflow"))) ||
		lowered.includes("dci_context_window_preflight")
	) {
		return makeFailure("context_overflow", message);
	}
	return makeFailure("agent_error", message);
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
			const stepUsage = parseUsage(step.usage);
			message.usage = stepUsage.available ? availableUsage(stepUsage.value) : unavailableUsage();
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

export function createBridgeStreamFn(args: Args): StreamFn {
	return async (model, context, options?: SimpleStreamOptions) => {
		const stream = createAssistantMessageEventStream();
		queueMicrotask(async () => {
			try {
				const baseUrl = args.openaiBaseUrl.replace(/\/+$/, "");
				const estimatedInputTokens = estimateNextInputTokens(context);
				if (
					estimatedInputTokens + DEFAULT_CONTRACT.max_model_output_tokens + CONTEXT_SAFETY_TOKENS >
					model.contextWindow
				) {
					throw new Error(
						`DCI_CONTEXT_WINDOW_PREFLIGHT: estimated input ${estimatedInputTokens} plus ` +
							`${DEFAULT_CONTRACT.max_model_output_tokens} output and ${CONTEXT_SAFETY_TOKENS} safety tokens ` +
							`exceeds model context window ${model.contextWindow}`,
					);
				}
				const readOnly = context.tools?.length === 1 && context.tools[0].name === "read";
				const request = {
					model: args.model,
					messages: toOpenAIMessages(context),
					tools: toOpenAITools(context.tools),
					tool_choice: options?.toolChoice === "none" ? "none" : readOnly ? { type: "function", function: { name: "read" } } : "auto",
					max_tokens: DEFAULT_CONTRACT.max_model_output_tokens,
					temperature: DEFAULT_CONTRACT.generation_config.temperature,
					seed: DEFAULT_CONTRACT.generation_config.seed,
					chat_template_kwargs: { enable_thinking: DEFAULT_CONTRACT.generation_config.enable_thinking },
				};
				const response = await fetch(`${baseUrl}/chat/completions`, {
					method: "POST",
					headers: {
						"content-type": "application/json",
						authorization: `Bearer ${args.openaiApiKey}`,
					},
					body: JSON.stringify(request),
					signal: options?.signal,
				});
				const payload = await response.json() as {
					usage?: unknown;
					choices?: Array<{ finish_reason?: string; message?: {
						content?: unknown;
						tool_calls?: Array<{ id?: string; function?: { name?: string; arguments?: string } }>;
					} }>;
				};
				if (!response.ok) throw new Error(JSON.stringify(payload));
				const choice = payload.choices?.[0];
				const output = choice?.message ?? {};
				const content: AssistantMessage["content"] = [];
				if (typeof output.content === "string" && output.content) {
					content.push({ type: "text", text: output.content });
				}
				const toolsDisabled = options?.toolChoice === "none";
				// A backend that ignores tool_choice must not execute a late tool call.
				// Preserve the response and its usage as an invalid final for repair.
				if (toolsDisabled && output.tool_calls?.length) {
					content.splice(0, content.length, { type: "text", text: JSON.stringify({ blocked_tool_calls: output.tool_calls }) });
				}
				for (const toolCall of toolsDisabled ? [] : output.tool_calls ?? []) {
					if (!toolCall.function?.name) throw new Error("backend returned a tool call without a name");
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
				const pseudoToolCall = !toolsDisabled && content.length === 1 && content[0]?.type === "text" ? parsePseudoToolCall(content[0].text) : undefined;
				if (pseudoToolCall) {
					content.splice(0, content.length, pseudoToolCall);
				}
				const stopReason = choice?.finish_reason === "length" ? "length" : content.some((part) => part.type === "toolCall") ? "toolUse" : "stop";
				const responseUsage = parseUsage(payload.usage);
				const message: AssistantMessage = {
					role: "assistant",
					content,
					api: model.api,
					provider: model.provider,
					model: model.id,
					usage: responseUsage.available ? availableUsage(responseUsage.value) : unavailableUsage(),
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
					usage: unavailableUsage(),
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

function buildPrompt(query: string, taskInstruction: string): string {
	const instruction = taskInstruction.trim();
	const instructionBlock = instruction
		? DEFAULT_CONTRACT.task_instruction_template.replace("{task_instruction}", instruction)
		: "";
	const prompt = DEFAULT_CONTRACT.query_prompt_template
		.replace("{query}", query)
		.replace("{task_instruction_block}", instructionBlock);
	return [prompt, DEFAULT_CONTRACT.budget_control?.initial_prompt, DEFAULT_CONTRACT.search_policy?.prompt]
		.filter((part) => part !== undefined).join("\n\n");
}

function buildRepairPrompt(reason: string, finalizing: boolean): string {
	const template = finalizing && DEFAULT_CONTRACT.budget_control
		? DEFAULT_CONTRACT.budget_control.final_repair_prompt
		: DEFAULT_CONTRACT.repair_prompt_template;
	return template.replace("{reason}", reason);
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
	const deadlineAt = Math.min(PROCESS_STARTED_AT + args.timeoutSeconds * 1000, args.deadlineAtMs ?? Infinity);
	const codeFiles = args.codeFiles ? parseCodeFiles(JSON.parse(await readFile(resolve(args.codeFiles), "utf8"))) : undefined;
	const sandbox = await validateCorpusSandbox(resolve(args.workspace), codeFiles);
	if (codeFiles) {
		for (const metadataPath of [args.codeFiles!, codeContractPath!]) {
			const canonical = await realpath(metadataPath);
			if (canonical === sandbox.corpus || canonical.startsWith(`${sandbox.corpus}${sep}`)) {
				throw new Error("Code runtime metadata must remain outside the tool-visible corpus");
			}
		}
	}
	await runSandboxPreflight(sandbox);
	if (args.preflightOnly) {
		console.log(`bubblewrap corpus sandbox preflight passed: ${sandbox.corpus}`);
		return;
	}

	const scratch = await mkdtemp(join(tmpdir(), "dci-bench-sample-"));
	let deadlineTimer: ReturnType<typeof setTimeout> | undefined;
	let onTerminate: (() => void) | undefined;
	const output = resolve(args.output);
	const trace = args.trace ? resolve(args.trace) : undefined;
	try {
		await mkdir(join(scratch, "tmp"), { mode: 0o700 });
		const query = await readFile(resolve(args.queryFile), "utf8");
		const taskInstruction = args.taskInstructionFile
			? (await readFile(resolve(args.taskInstructionFile), "utf8")).trim()
			: "";
		await mkdir(dirname(output), { recursive: true });
		if (trace) await mkdir(dirname(trace), { recursive: true });

		let turns = 0;
		let totalModelTokens = 0;
		let structuredFailure: StructuredFailure = null;
		const events: unknown[] = [];
		const evidence: EvidenceState = { bodyReadFiles: new Set(), contentSearchCommands: [], readReceipts: [] };
		const corpusIds = codeFiles ? undefined : corpusDocumentIds(await readdir(sandbox.corpus));
		const repairReasons: string[] = [];
		const budgetPolicy = DEFAULT_CONTRACT.budget_control;
		const controlEvents: Array<Record<string, unknown>> = [];
		let finalizing = false;
		let requiringRead = false;
		let requiredReadTurns = 0;
		let lastToolSignature = "";
		let identicalCalls = 0;
		let pendingControl: string | undefined;
		const model = createModel(args.modelProvider, args.model, args.openaiBaseUrl, args.contextWindow);
		const baseStreamFn =
			args.modelProvider === "mock" ? await createMockStreamFn(args.mockScript!) : createBridgeStreamFn(args);
		const readTool = codeFiles ? createCodeReadTool(sandbox, evidence) : createReadTool(sandbox, evidence);
		const searchTools = [readTool, createBashTool(sandbox, scratch, evidence)];
		const agent = new Agent({
			streamFn: (currentModel, context, options) =>
				baseStreamFn(currentModel, context, {
					...options, ...(finalizing ? { toolChoice: "none" as const } : {}),
				}),
			initialState: {
				systemPrompt: DEFAULT_CONTRACT.system_prompt,
				model,
				thinkingLevel: "off",
				tools: searchTools,
			},
			toolExecution: "sequential",
			shouldStopAfterTurn: async ({ message, context }) => {
				turns++;
				if (structuredFailure) return true;
				const currentUsage = readMessageUsage(message);
				if (!currentUsage.available) {
					structuredFailure = makeFailure(
						"usage_unavailable",
						`model usage was missing or incomplete on assistant turn ${turns}`,
					);
					return true;
				}
				totalModelTokens += currentUsage.value.input + currentUsage.value.output;
				if (Date.now() >= deadlineAt) {
					structuredFailure = makeFailure("timeout", "sample wall deadline reached before final acceptance");
					return true;
				}
				if (currentUsage.value.output > DEFAULT_CONTRACT.max_model_output_tokens) {
					structuredFailure = makeFailure("agent_error", "backend exceeded per-turn output token limit");
					return true;
				}
				if (totalModelTokens >= DEFAULT_CONTRACT.max_total_model_tokens) {
					structuredFailure = makeFailure(
						"token_limit",
						`cumulative model input/output tokens reached ${DEFAULT_CONTRACT.max_total_model_tokens}`,
					);
					return true;
				}
				const parsed = parseRuntimeFinal(finalAnswerFrom(context.messages), sandbox);
				if (parsed && hasBodyEvidence(evidence)) {
					// Any ID/schema repair after a completed retrieval must not restart searching.
					if (budgetPolicy) finalizing = true;
					return true;
				}
				if (turns >= args.maxAgentSteps) {
					structuredFailure = makeFailure(
						"max_agent_steps",
						"max_agent_steps reached before an acceptable final answer",
					);
					return true;
				}
				if (finalizing) return true;
				if (budgetPolicy) {
					const calls = message.content.filter((part): part is ToolCall => part.type === "toolCall");
					for (const call of calls) {
						const signature = JSON.stringify([call.name, sortJsonKeys(call.arguments)]);
						identicalCalls = signature === lastToolSignature ? identicalCalls + 1 : 1;
						lastToolSignature = signature;
					}
					const remaining = DEFAULT_CONTRACT.max_total_model_tokens - totalModelTokens;
					const forecast = estimateBudgetReserve({ ...context, messages: context.messages.filter((item): item is Message =>
						item.role === "assistant" || item.role === "user" || item.role === "toolResult") }, budgetPolicy);
					const { nextRequest, reserve } = forecast;
					const hasEvidence = hasBodyEvidence(evidence);
					const repeating = calls.length > 0 && identicalCalls >= budgetPolicy.repeat_threshold;
					const lowBudget = remaining <= reserve;
					const remainingSeconds = Math.max(0, (deadlineAt - Date.now()) / 1000);
					// Time is a hard timeout only. It must not alter the search policy
					// or turn a slow first request into a forced final answer.
					if (hasEvidence && requiringRead && !lowBudget) {
						requiringRead = false;
						requiredReadTurns = 0;
						identicalCalls = 0;
						lastToolSignature = "";
						pendingControl = budgetPolicy.resume_prompt.replace("{remaining_tokens}", String(remaining));
						controlEvents.push({ action: "resume_search", reason: "required corpus read completed with budget remaining",
							turn: turns, remaining_tokens: remaining, reserve_tokens: reserve, forecast,
							remaining_seconds: remainingSeconds,
							allowed_tools: ["read", "bash"], tool_choice: "auto" });
						return true;
					}
					if (hasEvidence && (repeating || lowBudget)) {
						const reason = repeating ? "identical tool calls repeated" : "remaining budget reserved for final submission";
						requiringRead = false;
						finalizing = true;
						pendingControl = budgetPolicy.final_prompt.replace("{reason}", reason).replace("{remaining_tokens}", String(remaining));
						controlEvents.push({ action: "finalize", reason, turn: turns, remaining_tokens: remaining, reserve_tokens: reserve, forecast,
							remaining_seconds: remainingSeconds, tool_choice: "none" });
						return true;
					}
					if (remaining <= nextRequest) {
						structuredFailure = makeFailure("token_limit", "remaining cumulative budget cannot safely fund another model request");
						return true;
					}
					if (requiringRead) {
						requiredReadTurns++;
						if (requiredReadTurns >= budgetPolicy.max_required_read_turns) {
							structuredFailure = makeFailure("tool_failure", "required read phase ended without a nonempty corpus read receipt");
						} else {
							pendingControl = budgetPolicy.read_prompt.replace("{remaining_tokens}", String(remaining));
						}
						return true;
					}
					if (!hasEvidence && (repeating || lowBudget || turns >= budgetPolicy.max_search_turns_without_read)) {
						requiringRead = true;
						pendingControl = budgetPolicy.read_prompt.replace("{remaining_tokens}", String(remaining));
						controlEvents.push({ action: "require_read", reason: repeating ? "repeated_tool" : lowBudget ? "low_budget" : "search_turn_limit",
							turn: turns, remaining_tokens: remaining, reserve_tokens: reserve, forecast, remaining_seconds: remainingSeconds,
							allowed_tools: ["read"], tool_choice: "read" });
						return true;
					}
				}
				return false;
			},
		});
		const startedAt = new Date().toISOString();
		onTerminate = () => {
			structuredFailure = makeFailure("timeout", "sample deadline or termination signal reached");
			agent.abort();
		};
		process.once("SIGTERM", onTerminate);
		process.once("SIGINT", onTerminate);
		const remainingMs = deadlineAt - Date.now();
		deadlineTimer = setTimeout(onTerminate, Math.max(1, remainingMs));
		const persistTrace = () => {
			if (!trace) return;
			const messages = agent.state.messages.filter((message): message is Message =>
				message.role === "assistant" || message.role === "user" || message.role === "toolResult");
			const observedTokens = messages.reduce((sum, message) => message.role === "assistant"
				? sum + readMessageUsage(message).value.totalTokens : sum, 0);
			writeJsonAtomicSync(trace, {
				...(codeFiles ? { evaluation_track: "code" } : {}),
				schema_version: DEFAULT_CONTRACT.trajectory_schema_version,
				contract_version: DEFAULT_CONTRACT.benchmark_contract_version,
				protocol_profile: PROTOCOL_PROFILE, base_contract_sha256: CONTRACT_SHA256, control_events: controlEvents,
				started_at: startedAt, updated_at: new Date().toISOString(), partial: true,
				inspect_sandbox: "local", tool_sandbox: "bubblewrap",
				task_instruction: taskInstruction, failure: structuredFailure,
				usage_summary: usageSummary(messages, observedTokens), usage_by_turn: usageByTurn(messages),
				events, messages, body_evidence: { read_receipts: evidence.readReceipts },
			});
		};
		persistTrace();
		agent.subscribe((event: AgentEvent) => {
			if (event.type === "turn_end" || event.type === "tool_execution_start" || event.type === "tool_execution_end") {
				events.push(event);
			}
			if (event.type !== "message_update") persistTrace();
		});
		const promptWithControls = async (prompt: string) => {
			let nextPrompt: string | undefined = prompt;
			while (nextPrompt && !structuredFailure) {
				pendingControl = undefined;
				if (Date.now() >= deadlineAt) {
					structuredFailure = makeFailure("timeout", "sample wall deadline reached before model request");
					break;
				}
				// Restrict execution as well as the advertised tools: a backend that
				// ignores the requested tool choice must not resume bash searches.
				agent.state.tools = finalizing ? [] : requiringRead ? [readTool] : searchTools;
				if (budgetPolicy && totalModelTokens > 0) {
					const messages = agent.state.messages.filter((item): item is Message =>
						item.role === "assistant" || item.role === "user" || item.role === "toolResult");
					const estimated = estimateNextInputTokens({ messages }, true) + Buffer.byteLength(nextPrompt, "utf8") +
						DEFAULT_CONTRACT.max_model_output_tokens + CONTEXT_SAFETY_TOKENS;
					if (estimated >= DEFAULT_CONTRACT.max_total_model_tokens - totalModelTokens) {
						structuredFailure = makeFailure("token_limit", "remaining cumulative budget cannot safely fund finalization or repair");
						break;
					}
				}
				try {
					await agent.prompt(nextPrompt);
				} catch (error) {
					structuredFailure ??= failureFromAgentError(String(error));
				}
				nextPrompt = pendingControl;
			}
		};
		await promptWithControls(buildPrompt(query, taskInstruction));
		const assistantMessages = agent.state.messages.filter(
			(message): message is AssistantMessage => message.role === "assistant",
		);
		const unavailableUsageMessage = assistantMessages.find((message) => !readMessageUsage(message).available);
		if (!structuredFailure && unavailableUsageMessage?.stopReason === "error") {
			structuredFailure = failureFromAgentError(unavailableUsageMessage.errorMessage ?? "agent returned an error");
		}
		if (!structuredFailure && unavailableUsageMessage) {
			structuredFailure = makeFailure("usage_unavailable", "model usage was missing or incomplete");
		}
		let finalText = finalAnswerFrom(agent.state.messages);
		let parsed = parseRuntimeFinal(finalText, sandbox);
		let finalIssue = runtimeFinalIssue(parsed, evidence, corpusIds);
		while (
			!structuredFailure &&
			finalIssue &&
			turns < args.maxAgentSteps &&
			repairReasons.length < DEFAULT_CONTRACT.max_final_repair_turns
		) {
			repairReasons.push(finalIssue);
			await promptWithControls(buildRepairPrompt(finalIssue, finalizing));
			const repairAssistantMessages = agent.state.messages.filter(
				(message): message is AssistantMessage => message.role === "assistant",
			);
			const repairUnavailableUsageMessage = repairAssistantMessages.find((message) => !readMessageUsage(message).available);
			if (!structuredFailure && repairUnavailableUsageMessage?.stopReason === "error") {
				structuredFailure = failureFromAgentError(
					repairUnavailableUsageMessage.errorMessage ?? "agent returned an error",
				);
			}
			if (!structuredFailure && repairUnavailableUsageMessage) {
				structuredFailure = makeFailure("usage_unavailable", "model usage was missing or incomplete");
			}
			finalText = finalAnswerFrom(agent.state.messages);
			parsed = parseRuntimeFinal(finalText, sandbox);
			finalIssue = runtimeFinalIssue(parsed, evidence, corpusIds);
		}
		const acceptedParsed = !structuredFailure && parsed !== undefined && finalIssue === undefined ? parsed : undefined;
		const toolCalls = agent.state.messages.filter((message) => message.role === "toolResult").length;
		const failure = structuredFailure ??
			(acceptedParsed
				? null
				: makeFailure("invalid_json", finalIssue ?? "final answer did not match the DCI ranked_doc_ids JSON schema"));
		const messageSnapshot = agent.state.messages.filter(
			(message): message is Message => message.role === "assistant" || message.role === "user" || message.role === "toolResult",
		);
		const usage = usageSummary(messageSnapshot, totalModelTokens);
		const completedAt = new Date().toISOString();
		const result = {
			pi_result_version: DEFAULT_CONTRACT.pi_final_result_version,
			contract_version: DEFAULT_CONTRACT.benchmark_contract_version,
			protocol_profile: PROTOCOL_PROFILE,
			base_contract_sha256: CONTRACT_SHA256,
			control_events: controlEvents,
			started_at: startedAt,
			completed_at: completedAt,
			valid_output: acceptedParsed !== undefined,
			failure_reason: failure?.message ?? null,
			failure,
			ranked_doc_ids: acceptedParsed && "ranked_doc_ids" in acceptedParsed ? acceptedParsed.ranked_doc_ids : [],
			...(codeFiles ? { evaluation_track: "code", code_output: acceptedParsed && "ranked_files" in acceptedParsed ? acceptedParsed : null } : {}),
			raw_final_text: finalText,
			agent_steps: turns,
			tool_calls: toolCalls,
			final_output_adapter: acceptedParsed && "adapter" in acceptedParsed ? acceptedParsed.adapter : null,
			repair_attempts: repairReasons.length,
			repair_reasons: repairReasons,
			usage_summary: usage,
			usage,
			body_evidence: {
				read_files: [...evidence.bodyReadFiles].sort(),
				content_search_commands: evidence.contentSearchCommands,
				read_receipts: evidence.readReceipts,
			},
			model_provider: model.provider,
			model: model.id,
			inspect_sandbox: "local",
			tool_sandbox: "bubblewrap",
			contract: {
				benchmark_contract_version: DEFAULT_CONTRACT.benchmark_contract_version,
				task_instruction: taskInstruction,
				top_k: DEFAULT_CONTRACT.top_k,
				max_agent_steps: args.maxAgentSteps,
				context_window: args.contextWindow,
				max_total_model_tokens: DEFAULT_CONTRACT.max_total_model_tokens,
				max_final_output_tokens: DEFAULT_CONTRACT.max_final_output_tokens,
				max_model_output_tokens: DEFAULT_CONTRACT.max_model_output_tokens,
				generation_config: DEFAULT_CONTRACT.generation_config,
				...(budgetPolicy ? { budget_control: budgetPolicy } : {}),
				...(DEFAULT_CONTRACT.search_policy ? { search_policy: DEFAULT_CONTRACT.search_policy } : {}),
				max_final_repair_turns: DEFAULT_CONTRACT.max_final_repair_turns,
				sample_timeout_seconds: args.timeoutSeconds,
				tool_result_max_bytes: DEFAULT_CONTRACT.tool_result_max_bytes,
				tool_allowlist: DEFAULT_CONTRACT.tool_allowlist,
				tool_sandbox: "bubblewrap",
				max_scratch_bytes: DEFAULT_CONTRACT.max_scratch_bytes,
			},
		};
		writeJsonAtomicSync(output, result);
		if (trace) {
			writeJsonAtomicSync(
				trace,
				{
						...(codeFiles ? { evaluation_track: "code" } : {}),
						schema_version: DEFAULT_CONTRACT.trajectory_schema_version,
						argv: redactArgv(process.argv.slice(2)),
						cwd: process.cwd(),
						workspace: sandbox.workspace,
						output,
						inspect_sandbox: "local",
						tool_sandbox: "bubblewrap",
						contract_version: DEFAULT_CONTRACT.benchmark_contract_version,
						protocol_profile: PROTOCOL_PROFILE,
						base_contract_sha256: CONTRACT_SHA256,
						control_events: controlEvents,
						task_instruction: taskInstruction,
						started_at: startedAt,
						completed_at: completedAt,
						failure,
						usage_summary: usage,
						usage,
						usage_by_turn: usageByTurn(messageSnapshot),
						body_evidence: result.body_evidence,
						events,
						messages: agent.state.messages,
					},
			);
		}
		if (!acceptedParsed) {
			console.error(`Invalid DCI final answer from ${basename(fileURLToPath(import.meta.url))}: ${finalText}`);
			process.exitCode = 2;
		}
	} finally {
		if (deadlineTimer) clearTimeout(deadlineTimer);
		if (onTerminate) { process.off("SIGTERM", onTerminate); process.off("SIGINT", onTerminate); }
		await rm(scratch, { recursive: true, force: true });
	}
}

if (resolve(process.argv[1] ?? "") === fileURLToPath(import.meta.url)) main().catch((error) => {
	console.error(error);
	process.exit(1);
});
