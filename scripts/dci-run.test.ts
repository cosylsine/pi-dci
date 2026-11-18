import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { chmod, cp, link, mkdtemp, mkdir, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { fauxAssistantMessage, type Context, type Model } from "@earendil-works/pi-ai";
import { corpusDocumentIds, createBridgeStreamFn, createReadTool, documentIdIssue, estimateBudgetReserve, parseFinalJson, parseTimeoutSeconds, parseUsage, sandboxArgs, truncate, utf8Prefix, type CorpusSandbox } from "./dci-run.ts";
import { parseAdaptedJson, parsePseudoToolCall, stripModelNoise } from "./adapters/inspect-bridge.ts";
import { parseRuntimeJson, validateRuntimeDescriptor, type FrozenRuntimeDescriptor } from "./dci-runtime.ts";

const PI_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DCI_RUN = join(dirname(fileURLToPath(import.meta.url)), "dci-run.ts");

type MockUsage = {
	input?: number;
	output?: number;
	cacheRead?: number;
	cacheWrite?: number;
	total_tokens?: number;
	totalTokens?: number;
};
type MockStep =
	| { type: "tool_call"; name: string; arguments: Record<string, unknown>; id?: string; usage?: MockUsage }
	| { type: "tool_calls"; calls: Array<{ name: string; arguments: Record<string, unknown>; id?: string }>; usage?: MockUsage }
	| { type: "final"; text: string; usage?: MockUsage };

type TraceEvent = { type?: string; toolName?: string; isError?: boolean };
type StageTiming = {
	sandbox_validate_seconds?: unknown;
	sandbox_preflight_seconds?: unknown;
	bash_wall_seconds?: unknown;
	bash_calls?: unknown;
};
type ControlEvent = { remaining_seconds?: unknown; action?: string; turn?: number; remaining_tokens?: number;
	forecast?: { finalReserve: number; requiredReadReserve: number; reserve: number } };
type RunnerResult = Record<string, unknown> & {
	failure?: { kind?: string } | null;
	contract?: { sample_timeout_seconds?: unknown };
	usage_summary?: {
		input_tokens?: unknown;
		output_tokens?: unknown;
		cache_read_tokens?: unknown;
		cache_write_tokens?: unknown;
		total_tokens?: unknown;
		total_model_tokens?: unknown;
	};
	stage_timing?: StageTiming;
	body_evidence?: { read_receipts?: unknown[] };
	agent_runtime?: Record<string, unknown>;
	trace?: {
		events?: TraceEvent[];
		stage_timing?: StageTiming;
		control_events?: ControlEvent[];
		usage_by_turn?: Array<Record<string, unknown>>;
		agent_runtime?: Record<string, unknown>;
	};
};

type FrozenRuntimeFixture = {
	root: string;
	rootfs: string;
	ready: string;
	descriptor: FrozenRuntimeDescriptor;
};

async function createFrozenRuntimeFixture(): Promise<FrozenRuntimeFixture> {
	const root = await mkdtemp(join(tmpdir(), "dci-frozen-runtime-test-"));
	const rootfs = join(root, "rootfs");
	const ready = join(root, "READY.json");
	await mkdir(join(rootfs, "usr/bin"), { recursive: true });
	for (const tool of ["bash", "rg"]) {
		const destination = join(rootfs, "usr/bin", tool);
		await cp(`/usr/bin/${tool}`, destination);
		await chmod(destination, 0o555);
	}
	await chmod(rootfs, 0o555);
	const hash = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
	const identity = {
		schema_version: "dci-agent-runtime-v1",
		mode: "frozen-rootfs",
		image_digest: `sha256:${"1".repeat(64)}`,
		platform: "linux/amd64",
		rootfs_tree_sha256: "2".repeat(64),
		tool_policy: "open",
		filesystem_policy: "dci-bwrap-frozen-ro-v1",
		runtime_policy_sha256: "3".repeat(64),
	} as const;
	await writeFile(ready, JSON.stringify({
		schema_version: "dci-rootfs-ready-v1",
		content_identity: {
			image_digest: identity.image_digest,
			platform: identity.platform,
			rootfs_tree_sha256: identity.rootfs_tree_sha256,
		},
		tools: {
			bash: { path: "usr/bin/bash", sha256: await hash(join(rootfs, "usr/bin/bash")), version: "fixture-bash" },
			rg: { path: "usr/bin/rg", sha256: await hash(join(rootfs, "usr/bin/rg")), version: "fixture-rg" },
		},
	}), { mode: 0o444 });
	return {
		root,
		rootfs,
		ready,
		descriptor: parseRuntimeJson(JSON.stringify({ identity, rootfs_path: rootfs })),
	};
}

async function runV2Mock(
	steps: MockStep[],
	tokenBudget?: string,
	timeoutSeconds: string | null = "30",
	runtime?: FrozenRuntimeDescriptor,
): Promise<RunnerResult> {
	const root = await mkdtemp(join(tmpdir(), "dci-v2-transcript-"));
	try {
		const workspace = join(root, "workspace");
		await mkdir(join(workspace, "corpus"), { recursive: true });
		await writeFile(join(workspace, "corpus", "doc.md"), "Relevant document body\n", "utf8");
		const query = join(root, "query.txt");
		const mock = join(root, "mock.json");
		const output = join(root, "result.json");
		const trace = join(root, "trace.json");
		await writeFile(query, "Find the relevant document", "utf8");
		await writeFile(mock, JSON.stringify(steps), "utf8");
		const environment: NodeJS.ProcessEnv = { ...process.env, DCI_PROTOCOL_PROFILE: "document-pool-v2-lite" };
		delete environment.DCI_PROTOCOL_CONTRACT_SHA256;
		if (tokenBudget === undefined) delete environment.DCI_DOCUMENT_TOKEN_BUDGET;
		else environment.DCI_DOCUMENT_TOKEN_BUDGET = tokenBudget;
		const timeoutArgs = timeoutSeconds === null ? [] : ["--timeout-seconds", timeoutSeconds];
		const runtimeArgs = runtime ? ["--runtime-json", JSON.stringify(runtime)] : [];
		const child = spawn(process.execPath, ["--import", "tsx", DCI_RUN,
			"--query-file", query, "--workspace", workspace, "--output", output,
			"--trace", trace, "--mock-script", mock, "--model", "mock", ...runtimeArgs, ...timeoutArgs],
			{ cwd: PI_ROOT, env: environment, stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		child.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString("utf8"); });
		child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString("utf8"); });
		const code = await new Promise<number>((resolve, reject) => {
			child.on("error", reject);
			child.on("close", (status) => resolve(status ?? -1));
		});
		assert.ok(code === 0 || code === 2, `unexpected runner exit ${code}: ${stderr || stdout}`);
		assert.ok(existsSync(output), `runner produced no result (${code}): ${stderr || stdout}`);
		const result = JSON.parse(await readFile(output, "utf8")) as RunnerResult;
		result.trace = JSON.parse(await readFile(trace, "utf8")) as RunnerResult["trace"];
		return result;
	} finally {
		await rm(root, { recursive: true, force: true });
	}
}

test("calibrated reserve uses observed usage and UTF-8 growth while preserving the v5 baseline", () => {
	const assistant = fauxAssistantMessage("tool call");
	assistant.usage = { ...assistant.usage, input: 5000, output: 100, totalTokens: 5100 };
	const context: Context = { messages: [
		{ role: "user", content: "already included in observed usage".repeat(1000), timestamp: 0 },
		assistant,
		{ role: "toolResult", toolCallId: "read", toolName: "read", content: [{ type: "text", text: "正文😀" }], isError: false, timestamp: 0 },
	] };
	const baseline = estimateBudgetReserve(context, { reserve_tokens: 32000 });
	const calibrated = estimateBudgetReserve(context, {
		reserve_tokens: 32000, forecast_bytes_per_token: 2, forecast_request_safety_tokens: 1024,
	});
	assert.equal(baseline.nextInput, 5110);
	assert.equal(baseline.reserve, 73762);
	assert.equal(calibrated.nextInput, 5105);
	assert.equal(calibrated.reserve, 44531);
	assert.ok(baseline.reserve > 60000 && calibrated.reserve < 60000);
	assert.equal(estimateBudgetReserve({ messages: [] }, {
		reserve_tokens: 32000, forecast_bytes_per_token: 2, forecast_request_safety_tokens: 1024,
	}).reserve, 32000);
	for (const ratio of [0, -1, Number.NaN]) {
		assert.throws(() => estimateBudgetReserve(context, { reserve_tokens: 32000, forecast_bytes_per_token: ratio }));
	}
});

test("timeout parser accepts unlimited aliases and preserves positive limits", () => {
	assert.equal(parseTimeoutSeconds(undefined, null, true), null);
	assert.equal(parseTimeoutSeconds("none", 600, true), null);
	assert.equal(parseTimeoutSeconds("NULL", 600, true), null);
	assert.equal(parseTimeoutSeconds("12.5", null, true), 12.5);
	assert.equal(parseTimeoutSeconds(undefined, 600, false), 600);
	assert.throws(() => parseTimeoutSeconds("none", 600, false), /must be positive$/);
	assert.throws(() => parseTimeoutSeconds(true, 600, true), /positive or none/);
	assert.throws(() => parseTimeoutSeconds("0", 600, true), /positive or none/);
});

test("usage normalization keeps explicit logical totals and rejects invalid totals", () => {
	const cached = parseUsage({
		prompt_tokens: 200,
		completion_tokens: 100,
		total_tokens: 1100,
		prompt_tokens_details: { cached_tokens: 800 },
	});
	assert.equal(cached.available, true);
	assert.equal(cached.value.input + cached.value.output, cached.value.totalTokens);
	assert.deepEqual(cached.value, {
		input: 1000,
		output: 100,
		cacheRead: 800,
		cacheWrite: 0,
		totalTokens: 1100,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	});
	assert.equal(parseUsage({ prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 }).value.totalTokens, 12);
	assert.equal(parseUsage({ prompt_tokens: 10, completion_tokens: 2 }).value.totalTokens, 12);
	for (const total_tokens of [null, "1100", -1, 11, 1.5, Number.NaN]) {
		assert.equal(parseUsage({ prompt_tokens: 10, completion_tokens: 2, total_tokens }).available, false);
	}
});

test("read accepts exact corpus filenames without decoding or opening paths outside corpus", async () => {
	const workspace = await mkdtemp(join(tmpdir(), "dci-path-test-"));
	try {
		const corpus = join(workspace, "corpus");
		await mkdir(corpus);
		await writeFile(join(corpus, "report.md.md"), "Inside corpus");
		await writeFile(join(workspace, "report.md.md"), "Outside corpus");
		await writeFile(join(corpus, "a%252Fb.md"), "Literal encoded filename");
		await writeFile(join(corpus, "%2E%2E%2Foutside.md"), "Literal traversal-like ID");
		await writeFile(join(workspace, "outside.md"), "Outside secret");
		await symlink(join(workspace, "outside.md"), join(corpus, "symlink.md"));
		await link(join(workspace, "outside.md"), join(corpus, "hardlink.md"));
		const evidence = { bodyReadFiles: new Set<string>(), contentSearchCommands: [], readReceipts: [] };
		const tool = createReadTool({ workspace, corpus, bwrap: "/unused" }, evidence);
		for (const path of ["report.md.md", "./report.md.md", "corpus/report.md.md", "/workspace/corpus/report.md.md"]) {
			const result = await tool.execute("read", { path });
			assert.deepEqual(result.content, [{ type: "text", text: '{"doc_id":"report.md"}\nInside corpus' }]);
		}
		for (const path of ["../outside.md", "corpus/../../outside.md", "/workspace/corpus/../outside.md", join(workspace, "outside.md")]) {
			await assert.rejects(tool.execute("read", { path }), /Path escapes corpus/);
		}
		await assert.rejects(tool.execute("read", { path: "symlink.md" }), /Symlinks are not allowed/);
		await assert.rejects(tool.execute("read", { path: "hardlink.md" }), /without hard links/);
		await assert.rejects(tool.execute("read", { path: "missing.md" }), /Copy the exact filename/);
		await tool.execute("read", { path: "a%252Fb.md" });
		await tool.execute("read", { path: "%2E%2E%2Foutside.md" });
		assert.equal(evidence.readReceipts.length, 6);
	} finally {
		await rm(workspace, { recursive: true, force: true });
	}
});

test("final parsing preserves exact original IDs, including percent signs and suffixes", () => {
	for (const id of ["report.md", "a%2Fb", "100%", ".md", "路径/😀", "<think>literal</think>"]) {
		assert.deepEqual(parseFinalJson(JSON.stringify({ ranked_doc_ids: [id] }))?.ranked_doc_ids, [id]);
	}
	for (const value of [[], [""], ["x", "x"], Array(11).fill("x"), [1]]) {
		assert.equal(parseFinalJson(JSON.stringify({ ranked_doc_ids: value })), undefined);
	}
	assert.equal(parseFinalJson('{"ranked_doc_ids":["x"],"extra":1}'), undefined);
	assert.equal(parseFinalJson('{"top_documents":["report.md"]}')?.adapter, "top_documents_to_ranked_doc_ids");
	assert.deepEqual(parseFinalJson('<think>reasoning</think>{"ranked_doc_ids":["<think>literal</think>"]}')?.ranked_doc_ids, ["<think>literal</think>"]);
});

test("inspect bridge adapter strips only outer model noise and recognizes pseudo tool calls", () => {
	const source = '<think>reasoning</think>\n{"ranked_doc_ids":["<think>literal</think>"]}';
	assert.equal(stripModelNoise(source), '{"ranked_doc_ids":["<think>literal</think>"]}');
	assert.deepEqual(parseAdaptedJson(source), { ranked_doc_ids: ["<think>literal</think>"] });
	const call = parsePseudoToolCall('<think>reasoning</think>{"name":"read","arguments":{"path":"doc.md"}}');
	assert.equal(call?.type, "toolCall");
	assert.equal(call?.name, "read");
	assert.deepEqual(call?.arguments, { path: "doc.md" });
	assert.equal(parsePseudoToolCall('{"name":"write","arguments":{}}'), undefined);
});

test("UTF-8 output cap counts bytes and never slices a code point", () => {
	assert.equal(utf8Prefix("a😀é", 5), "a😀");
	assert.equal(utf8Prefix("a😀é", 4), "a");
	for (const text of ["a".repeat(40000), "é".repeat(20000), "中文😀".repeat(9000)]) {
		const output = truncate(text);
		assert.ok(Buffer.byteLength(output) <= 20000);
		assert.ok(output.endsWith("\n[truncated]"));
		assert.ok(!output.includes("�"));
	}
});

test("ID membership preserves canonical suffixes and percent escapes, and diagnoses filename aliases", () => {
	const ids = corpusDocumentIds(["report.md.md", "a%252Fb.md", "100%25.md", "l%27exemple.md", "doc.md"]);
	for (const id of ["report.md", "a%2Fb", "100%", "l'exemple", "doc"]) {
		assert.equal(documentIdIssue({ ranked_doc_ids: [id] }, ids), undefined);
	}
	for (const [alias, id] of [["report.md.md", "report.md"], ["a%252Fb", "a%2Fb"], ["l%27exemple", "l'exemple"], ["doc.md", "doc"]]) {
		assert.ok(documentIdIssue({ ranked_doc_ids: [alias] }, ids)?.includes(JSON.stringify({ submitted: alias, doc_id: id })));
	}
	assert.ok(documentIdIssue({ ranked_doc_ids: ["invented"] }, ids)?.includes("Unknown original document IDs"));
	assert.throws(() => corpusDocumentIds(["invalid%.md"]));
	const collision = corpusDocumentIds(["doc.md", "doc.md.md"]);
	assert.equal(documentIdIssue({ ranked_doc_ids: ["doc.md"] }, collision), undefined);
});

test("only visible nonempty reads create receipts; offsets count Unicode code points", async () => {
	const workspace = await mkdtemp(join(tmpdir(), "dci-read-test-"));
	try {
		const corpus = join(workspace, "corpus");
		await mkdir(corpus);
		await writeFile(join(corpus, "doc.md"), "A😀é正文");
		await writeFile(join(corpus, "blank.md"), " ".repeat(30000) + "hidden");
		const evidence = { bodyReadFiles: new Set<string>(), contentSearchCommands: [], readReceipts: [] };
		const tool = createReadTool({ workspace, corpus, bwrap: "/unused" }, evidence);
		await tool.execute("directory", { path: "/workspace/corpus" });
		await tool.execute("beyond-eof", { path: "/workspace/corpus/doc.md", offset: 99 });
		await tool.execute("blank", { path: "/workspace/corpus/blank.md", limit: 40000 });
		assert.equal(evidence.readReceipts.length, 0);
		const result = await tool.execute("content", { path: "/workspace/corpus/doc.md", offset: 1, limit: 2 });
		assert.deepEqual(result.content, [{ type: "text", text: '{"doc_id":"doc"}\n😀é' }]);
		assert.deepEqual(evidence.readReceipts, [{
			path: "/workspace/corpus/doc.md", doc_id: "doc", offset_codepoints: 1, returned_codepoints: 2,
			returned_bytes: 6, content_sha256: createHash("sha256").update("😀é").digest("hex"),
		}]);
		const capped = await tool.execute("capped", { path: "/workspace/corpus/blank.md", limit: 40000 });
		assert.ok(Buffer.byteLength(capped.content[0].type === "text" ? capped.content[0].text : "") <= 20000);
		assert.equal(evidence.readReceipts.length, 1); // ID metadata alone is not body evidence.
	} finally {
		await rm(workspace, { recursive: true, force: true });
	}
});

test("bridge sends canonical generation parameters, preserves length stop, and cancels HTTP", async () => {
	const requests: Record<string, unknown>[] = [];
	let holdRequest = false;
	const server = createServer(async (request, response) => {
		const chunks: Buffer[] = [];
		for await (const chunk of request) chunks.push(Buffer.from(chunk));
		requests.push(JSON.parse(Buffer.concat(chunks).toString("utf8")));
		if (holdRequest) return;
		response.setHeader("content-type", "application/json");
		response.end(JSON.stringify({ choices: [{ message: { content: "truncated" }, finish_reason: "length" }], usage: { prompt_tokens: 10, completion_tokens: 2 } }));
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	try {
		const address = server.address();
		assert.ok(address && typeof address !== "string");
		const args = {
			preflightOnly: false, queryFile: "", workspace: "", output: "", modelProvider: "bridge" as const,
			openaiBaseUrl: `http://127.0.0.1:${address.port}/v1`, openaiApiKey: "test", model: "mock",
			maxAgentSteps: 10, contextWindow: 65536, timeoutSeconds: 10,
		};
		const model: Model<"openai-completions"> = { id: "mock", name: "mock", baseUrl: args.openaiBaseUrl, provider: "mock", api: "openai-completions", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 65536, maxTokens: 1024 };
		const streamFn = createBridgeStreamFn({
			baseUrl: args.openaiBaseUrl, apiKey: args.openaiApiKey, model: args.model,
			maxTokens: 1024, temperature: 0, seed: 0, enableThinking: false,
		});
		const stream = await streamFn(model, { messages: [] }, { signal: AbortSignal.timeout(5000) });
		assert.equal((await stream.result()).stopReason, "length");
		assert.equal(requests[0].max_tokens, 1024);
		assert.equal(requests[0].temperature, 0);
		assert.equal(requests[0].seed, 0);
		holdRequest = true;
		const controller = new AbortController();
		const pending = await streamFn(model, { messages: [] }, { signal: controller.signal });
		setTimeout(() => controller.abort(), 100);
		assert.equal((await pending.result()).stopReason, "error");
	} finally {
		server.closeAllConnections();
		await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
	}
});

test("bridge adapter blocks disabled tools and normalizes pseudo tool responses", async () => {
	const responses = [
		{
			choices: [{ message: {
				content: "ignored",
				tool_calls: [{ id: "late", function: { name: "read", arguments: '{"path":"doc.md"}' } }],
			}, finish_reason: "length" }],
			usage: { prompt_tokens: 10, completion_tokens: 2 },
		},
		{
			choices: [{ message: { content: '<think>reasoning</think>{"name":"read","arguments":{"path":"doc.md"}}' }, finish_reason: "stop" }],
			usage: { prompt_tokens: 10, completion_tokens: 2 },
		},
		{
			choices: [{ message: { content: '<tool_call><function=read><parameter=path>doc.md</parameter></function></tool_call>' }, finish_reason: "stop" }],
			usage: { prompt_tokens: 10, completion_tokens: 2 },
		},
	];
	let responseIndex = 0;
	const requests: Record<string, unknown>[] = [];
	const server = createServer(async (request, response) => {
		const chunks: Buffer[] = [];
		for await (const chunk of request) chunks.push(Buffer.from(chunk));
		requests.push(JSON.parse(Buffer.concat(chunks).toString("utf8")));
		response.setHeader("content-type", "application/json");
		response.end(JSON.stringify(responses[responseIndex++]));
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	try {
		const address = server.address();
		assert.ok(address && typeof address !== "string");
		const config = {
			baseUrl: `http://127.0.0.1:${address.port}/v1`, apiKey: "test", model: "mock",
			maxTokens: 1024, temperature: 0, seed: 0, enableThinking: false,
		};
		const model: Model<"openai-completions"> = {
			id: "mock", name: "mock", baseUrl: config.baseUrl, provider: "mock", api: "openai-completions",
			reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 65536, maxTokens: 1024,
		};
		const disabled = await createBridgeStreamFn(config)(model, { messages: [] }, { toolChoice: "none" });
		const disabledMessage = await disabled.result();
		assert.equal(disabledMessage.stopReason, "length");
		assert.equal(disabledMessage.content.find((part) => part.type === "toolCall"), undefined);
		assert.match(disabledMessage.content[0]?.type === "text" ? disabledMessage.content[0].text : "", /blocked_tool_calls/);

		const pseudo = await createBridgeStreamFn(config)(model, { messages: [] });
		const pseudoMessage = await pseudo.result();
		assert.equal(pseudoMessage.stopReason, "toolUse");
		const pseudoTool = pseudoMessage.content.find((part) => part.type === "toolCall");
		assert.equal(pseudoTool?.type, "toolCall");
		if (pseudoTool?.type === "toolCall") assert.deepEqual(pseudoTool.arguments, { path: "doc.md" });

		const xml = await createBridgeStreamFn(config)(model, { messages: [] }, { toolChoice: "none" });
		const xmlMessage = await xml.result();
		assert.equal(xmlMessage.content.find((part) => part.type === "toolCall"), undefined);
		assert.match(xmlMessage.content[0]?.type === "text" ? xmlMessage.content[0].text : "", /<tool_call>/);
		for (const index of [0, 2]) {
			assert.equal(requests[index].tool_choice, "none");
			assert.deepEqual(requests[index].tools, []);
		}
	} finally {
		server.closeAllConnections();
		await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
	}
});

test("bridge preserves Inspect cached prompt totals in the assistant usage", async () => {
	const server = createServer(async (_request, response) => {
		response.setHeader("content-type", "application/json");
		response.end(JSON.stringify({
			choices: [{ message: { content: "cached response" }, finish_reason: "stop" }],
			usage: { prompt_tokens: 200, completion_tokens: 100, total_tokens: 1100, prompt_tokens_details: { cached_tokens: 800 } },
		}));
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	try {
		const address = server.address();
		assert.ok(address && typeof address !== "string");
		const args = {
			preflightOnly: false, queryFile: "", workspace: "", output: "", modelProvider: "bridge" as const,
			openaiBaseUrl: `http://127.0.0.1:${address.port}/v1`, openaiApiKey: "test", model: "mock",
			maxAgentSteps: 10, contextWindow: 65536, timeoutSeconds: 10,
		};
		const model: Model<"openai-completions"> = {
			id: "mock", name: "mock", baseUrl: args.openaiBaseUrl, provider: "mock", api: "openai-completions",
			reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 65536, maxTokens: 1024,
		};
		const stream = await createBridgeStreamFn({
			baseUrl: args.openaiBaseUrl, apiKey: args.openaiApiKey, model: args.model,
			maxTokens: 1024, temperature: 0, seed: 0, enableThinking: false,
		})(
			model,
			{ messages: [] },
			{ signal: AbortSignal.timeout(5000) },
		);
		const message = await stream.result();
		assert.equal(message.usage.input, 1000);
		assert.equal(message.usage.output, 100);
		assert.equal(message.usage.cacheRead, 800);
		assert.equal(message.usage.totalTokens, 1100);
	} finally {
		server.closeAllConnections();
		await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
	}
});

test("explicit totals drive context estimates and cumulative cross-turn budget", async () => {
	const cached = fauxAssistantMessage("cached response");
	cached.usage = {
		...cached.usage,
		input: 1000,
		output: 100,
		cacheRead: 800,
		totalTokens: 1100,
	};
	assert.equal(estimateBudgetReserve({ messages: [cached] }, { reserve_tokens: 0 }).nextInput, 1100);

	const result = await runV2Mock([
		{
			type: "tool_call",
			name: "read",
			arguments: { path: "doc.md" },
			usage: { input: 100, output: 1, cacheRead: 11899, total_tokens: 12000 },
		},
		{
			type: "final",
			text: JSON.stringify({ ranked_doc_ids: ["doc"] }),
			usage: { input: 100, output: 1, cacheRead: 19899, total_tokens: 20000 },
		},
	], "32000");
	assert.equal(result.valid_output, false);
	assert.equal(result.failure?.kind, "token_limit");
	assert.equal(result.usage_summary?.input_tokens, 31998);
	assert.equal(result.usage_summary?.output_tokens, 2);
	assert.equal(result.usage_summary?.total_tokens, 32000);
	assert.equal(result.usage_summary?.total_model_tokens, 32000);
	assert.deepEqual(result.trace?.usage_by_turn?.map((turn) => turn.total_tokens), [12000, 20000]);
});

test("frozen runtime validates the descriptor, READY identity, and tool hashes", {
	skip: !existsSync("/usr/bin/bash") || !existsSync("/usr/bin/rg"),
}, async () => {
	const fixture = await createFrozenRuntimeFixture();
	try {
		const validated = validateRuntimeDescriptor(fixture.descriptor);
		assert.equal(validated.rootfs_path, await realpath(fixture.rootfs));
		const ready = JSON.parse(await readFile(fixture.ready, "utf8")) as Record<string, unknown>;
		const readyIdentity = ready.content_identity as Record<string, unknown>;
		await chmod(fixture.rootfs, 0o755);
		assert.throws(() => validateRuntimeDescriptor(fixture.descriptor), /must be read-only/);
		await chmod(fixture.rootfs, 0o555);
		await chmod(fixture.ready, 0o644);
		assert.throws(() => validateRuntimeDescriptor(fixture.descriptor), /READY.json must be read-only/);
		await chmod(fixture.ready, 0o444);
		readyIdentity.rootfs_tree_sha256 = "4".repeat(64);
		await writeFile(fixture.ready, JSON.stringify(ready));
		assert.throws(() => validateRuntimeDescriptor(fixture.descriptor), /content_identity does not match/);
		readyIdentity.rootfs_tree_sha256 = fixture.descriptor.identity.rootfs_tree_sha256;
		await writeFile(fixture.ready, JSON.stringify(ready));
		await writeFile(join(fixture.rootfs, "usr/bin/rg"), Buffer.concat([
			await readFile(join(fixture.rootfs, "usr/bin/rg")), Buffer.from("\0"),
		]));
		assert.throws(() => validateRuntimeDescriptor(fixture.descriptor), /hash does not match/);
	} finally {
		await rm(fixture.root, { recursive: true, force: true });
	}
});

test("frozen runtime rejects malformed identity and non-absolute rootfs paths", () => {
	const identity = {
		schema_version: "dci-agent-runtime-v1",
		mode: "host-tools",
		image_digest: `sha256:${"1".repeat(64)}`,
		platform: "linux/amd64",
		rootfs_tree_sha256: "2".repeat(64),
		tool_policy: "open",
		filesystem_policy: "dci-bwrap-frozen-ro-v1",
		runtime_policy_sha256: "3".repeat(64),
	};
	assert.throws(() => parseRuntimeJson(JSON.stringify({ identity, rootfs_path: "/tmp/rootfs" })), /identity.mode/);
	identity.mode = "frozen-rootfs";
	assert.throws(() => parseRuntimeJson(JSON.stringify({ identity, rootfs_path: "relative/rootfs" })), /absolute/);
});

test("frozen sandbox mounts the rootfs and scratch tmp while host mode keeps legacy mounts", {
	skip: !existsSync("/usr/bin/bash") || !existsSync("/usr/bin/rg"),
}, async () => {
	const fixture = await createFrozenRuntimeFixture();
	try {
		const validated = validateRuntimeDescriptor(fixture.descriptor);
		const scratch = join(fixture.root, "scratch");
		const baseSandbox = {
			workspace: join(fixture.root, "workspace"),
			corpus: join(fixture.root, "workspace/corpus"),
			bwrap: "/usr/bin/bwrap",
			rg: "/usr/bin/rg",
		} satisfies CorpusSandbox;
		const frozen = sandboxArgs({ ...baseSandbox, runtime: validated }, scratch, "true");
		const frozenRootBind = frozen.indexOf("--ro-bind");
		assert.equal(frozen[frozenRootBind + 1], validated.rootfs_path);
		assert.equal(frozen[frozenRootBind + 2], "/");
		assert.ok(!frozen.some((value, index) => value === "/usr/bin" && frozen[index - 1] === "--ro-bind"));
		assert.ok(frozen.includes(join(scratch, "tmp")));
		assert.equal(frozen[frozen.indexOf("TMPDIR") + 1], "/tmp");
		assert.equal(frozen[frozen.indexOf("PATH") + 1], "/usr/bin:/bin");

		const host = sandboxArgs({ ...baseSandbox, runtime: undefined }, scratch, "true");
		assert.ok(host.some((value, index) => value === "/usr/bin" && host[index - 1] === "--ro-bind"));
		assert.ok(host.some((value, index) => value === "/usr/bin/rg" && host[index - 1] === "--ro-bind"));
		assert.equal(host[host.indexOf("TMPDIR") + 1], "/scratch/tmp");
	} finally {
		await rm(fixture.root, { recursive: true, force: true });
	}
});

test("v2 frozen rootfs runs rootfs tools and preserves read evidence", {
	skip: !process.env.DCI_TEST_FROZEN_RUNTIME_JSON || !existsSync("/usr/bin/bwrap"),
}, async () => {
	const runtime = parseRuntimeJson(process.env.DCI_TEST_FROZEN_RUNTIME_JSON!);
	const result = await runV2Mock([
		{ type: "tool_call", name: "bash", arguments: { command: "rg -q Relevant /workspace/corpus/doc.md && touch /tmp/probe && test -f /scratch/tmp/probe" }, usage: { input: 10, output: 1 } },
		{ type: "tool_call", name: "read", arguments: { path: "doc.md" }, usage: { input: 10, output: 1 } },
		{ type: "final", text: JSON.stringify({ ranked_doc_ids: ["doc"] }), usage: { input: 10, output: 1 } },
	], undefined, "30", runtime);
	assert.equal(result.valid_output, true);
	assert.deepEqual(result.agent_runtime, runtime.identity);
	assert.deepEqual(result.trace?.agent_runtime, runtime.identity);
	assert.equal(result.body_evidence?.read_receipts?.length, 1);
});

test("v2 looks ahead through another search, tool growth, final and repair prompts", () => {
	const assistant = fauxAssistantMessage("read");
	assistant.usage = { ...assistant.usage, input: 5000, output: 100, totalTokens: 5100 };
	const context: Context = { messages: [
		assistant,
		{ role: "toolResult", toolCallId: "read", toolName: "read", content: [{ type: "text", text: "正文😀" }], isError: false, timestamp: 0 },
	] };
	const policy = { reserve_tokens: 0, dynamic_reserve: true,
		final_prompt: "final 正文", final_repair_prompt: "repair", read_prompt: "read" };
	const forecast = estimateBudgetReserve(context, policy);
	assert.equal(forecast.nextInput, 5110);
	const final = 5110 + Buffer.byteLength(policy.final_prompt) + 1024 + 4096;
	const repair = final + 1024 + Buffer.byteLength(policy.final_repair_prompt);
	assert.equal(forecast.finalReserve, final + repair);
	assert.equal(forecast.reserve, forecast.nextRequest + final + repair + 2 * (1024 + forecast.toolGrowth));
	// The old condition allowed a search here although its subsequent final/repair could not fit.
	assert.ok(forecast.finalReserve! < 36325);
	assert.ok(forecast.reserve > 36325);
	const unread = estimateBudgetReserve(context, policy, { needsRead: true });
	assert.equal(unread.reserve, forecast.nextRequest + unread.requiredReadReserve! + 3 * (1024 + forecast.toolGrowth));
	assert.ok(unread.reserve > forecast.reserve);
	const resumed = estimateBudgetReserve(context, policy, { continuationPrompt: "resume" });
	assert.equal(resumed.reserve - forecast.reserve, 3 * Buffer.byteLength("resume"));
	const diagnostic = documentIdIssue({ ranked_doc_ids: ["unknown".repeat(1000)] }, corpusDocumentIds(["doc.md"]))!;
	assert.equal(Buffer.byteLength(diagnostic), 4000);
	const rendered = estimateBudgetReserve(context, { ...policy,
		final_prompt: "final {reason}", final_repair_prompt: "repair {reason}", read_prompt: "read {remaining_tokens}" });
	const finalBytes = Buffer.byteLength("final remaining budget reserved for final submission");
	assert.equal(rendered.finalReserve, 2 * (5110 + finalBytes + 5120) + 1024 + Buffer.byteLength(`repair ${diagnostic}`));
});

test("v2 finalizes before an extra search and repairs XML without executing it", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const result = await runV2Mock([
		{ type: "tool_call", name: "read", arguments: { path: "doc.md", limit: 3 }, usage: { input: 12000, output: 1 } },
		{ type: "tool_call", name: "read", arguments: { path: "doc.md" }, usage: { input: 13000, output: 1 } },
		{ type: "final", text: '<tool_call><function=bash><parameter=command>echo unwanted</parameter></function></tool_call>', usage: { input: 14000, output: 100 } },
		{ type: "final", text: '{"ranked_doc_ids":["doc"]}', usage: { input: 15000, output: 10 } },
	], "96000");
	assert.equal(result.valid_output, true);
	assert.equal(result.agent_steps, 4);
	assert.equal(result.repair_attempts, 1);
	assert.equal(result.tool_invocations, 2);
	assert.equal(result.trace?.events?.filter((event) => event.type === "tool_execution_start" && event.toolName === "bash").length, 0);
	const final = result.trace?.control_events?.find((event) => event.action === "finalize");
	assert.equal(final?.turn, 2);
	assert.ok(final!.remaining_tokens! > final!.forecast!.finalReserve);
	assert.ok(final!.remaining_tokens! < final!.forecast!.reserve);
});

test("v2 requests evidence early and can finalize plus repair inside 32k", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const result = await runV2Mock([
		{ type: "tool_call", name: "bash", arguments: { command: "ls /workspace/corpus" }, usage: { input: 2000, output: 10 } },
		{ type: "tool_call", name: "read", arguments: { path: "doc.md" }, usage: { input: 3000, output: 10 } },
		{ type: "final", text: "not-json", usage: { input: 4000, output: 10 } },
		{ type: "final", text: '{"ranked_doc_ids":["doc"]}', usage: { input: 5000, output: 10 } },
	], "32000");
	assert.equal(result.valid_output, true);
	assert.equal(result.repair_attempts, 1);
	assert.equal(result.usage_summary?.total_model_tokens, 14040);
	assert.deepEqual(result.trace?.control_events?.map((event) => [event.action, event.turn]), [["require_read", 1], ["finalize", 2]]);
});

test("v2 does not count a repair blocked by the pre-dispatch token guard", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const result = await runV2Mock([
		{ type: "tool_call", name: "read", arguments: { path: "doc.md" }, usage: { input: 12000, output: 1 } },
		{ type: "final", text: "not-json", usage: { input: 10000, output: 100 } },
	], "32000");
	assert.equal(result.valid_output, false);
	assert.equal(result.failure?.kind, "token_limit");
	assert.equal(result.agent_steps, 2);
	assert.equal(result.repair_attempts, 0);
	assert.deepEqual(result.repair_reasons, []);
});

test("v2 stops after twelve search assistant turns and preserves a final turn", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const steps: MockStep[] = [];
	for (let offset = 0; offset < 12; offset++) {
		steps.push({ type: "tool_call", name: "read", arguments: { path: "doc.md", offset, limit: 4 }, usage: { input: 10, output: 1 } });
	}
	steps.push({ type: "final", text: JSON.stringify({ ranked_doc_ids: ["doc"] }), usage: { input: 10, output: 1 } });
	const result = await runV2Mock(steps);
	assert.equal(result.valid_output, true);
	assert.equal(result.search_turns, 12);
	assert.equal(result.tool_invocations, 12);
	assert.equal(result.agent_steps, 13);
	assert.equal(result.repair_attempts, 0);
});

test("v2 accepts an unlimited sample timeout and records no wall-clock remainder", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const steps: MockStep[] = [];
	for (let offset = 0; offset < 12; offset++) {
		steps.push({ type: "tool_call", name: "read", arguments: { path: "doc.md", offset, limit: 4 }, usage: { input: 10, output: 1 } });
	}
	steps.push({ type: "final", text: JSON.stringify({ ranked_doc_ids: ["doc"] }), usage: { input: 10, output: 1 } });
	for (const timeoutSeconds of [null, "none", "null"] as const) {
		const result = await runV2Mock(steps, undefined, timeoutSeconds);
		assert.equal(result.valid_output, true);
		assert.equal(result.contract?.sample_timeout_seconds, null);
		const controlEvents = result.trace?.control_events ?? [];
		assert.ok(controlEvents.length > 0);
		assert.ok(controlEvents.every((event) => event.remaining_seconds === null));
	}
});

test("v2 caps parallel tool arrays before executing an extra call", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const calls = Array.from({ length: 25 }, (_, index) => ({
		name: "bash", id: `bash-${index}`, arguments: { command: "true" },
	}));
	const result = await runV2Mock([
		{ type: "tool_calls", calls, usage: { input: 10, output: 1 } },
	]);
	assert.equal(result.valid_output, false);
	assert.equal(result.failure?.kind, "tool_failure");
	assert.equal(result.tool_invocations, 24);
	assert.equal(result.tool_calls, 24);
	assert.equal(result.search_turns, 1);
	const executedBash = result.trace?.events?.filter((event) =>
		event.type === "tool_execution_end" && event.toolName === "bash" && event.isError === false).length ?? 0;
	assert.equal(executedBash, 24);
});

test("v2 reports monotonic sandbox and bash stage timing", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const result = await runV2Mock([
		{ type: "tool_call", name: "bash", arguments: { command: "true" }, usage: { input: 10, output: 1 } },
		{ type: "tool_call", name: "bash", arguments: { command: "true" }, usage: { input: 10, output: 1 } },
		{ type: "tool_call", name: "read", arguments: { path: "doc.md" }, usage: { input: 10, output: 1 } },
		{ type: "final", text: JSON.stringify({ ranked_doc_ids: ["doc"] }), usage: { input: 10, output: 1 } },
	], "96000");
	assert.equal(result.valid_output, true);
	const timing = result.stage_timing;
	assert.ok(timing);
	for (const field of ["sandbox_validate_seconds", "sandbox_preflight_seconds", "bash_wall_seconds"] as const) {
		assert.equal(typeof timing[field], "number");
		assert.ok((timing[field] as number) >= 0);
	}
	assert.equal(timing.bash_calls, 2);
	assert.deepEqual(result.trace?.stage_timing, timing);
});

test("v2 required read consumes one of twelve search turns before finalization", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const steps: MockStep[] = [];
	for (let index = 0; index < 6; index++) {
		steps.push({ type: "tool_call", name: "bash", arguments: { command: `true # search-${index}` }, usage: { input: 10, output: 1 } });
	}
	steps.push({ type: "tool_call", name: "read", arguments: { path: "doc.md" }, usage: { input: 10, output: 1 } });
	for (let index = 0; index < 5; index++) {
		steps.push({ type: "tool_call", name: "bash", arguments: { command: `true # post-read-${index}` }, usage: { input: 10, output: 1 } });
	}
	steps.push({ type: "final", text: JSON.stringify({ ranked_doc_ids: ["doc"] }), usage: { input: 10, output: 1 } });
	// Leave enough budget to exercise the turn cap rather than early budget finalization.
	const result = await runV2Mock(steps, "96000");
	assert.equal(result.valid_output, true);
	assert.equal(result.search_turns, 12);
	assert.equal(result.tool_invocations, 12);
	assert.equal(result.agent_steps, 13);
	assert.equal(result.repair_attempts, 0);
});

test("v2 permits exactly one final repair after a completed read phase", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const read: MockStep = { type: "tool_call", name: "read", arguments: { path: "doc.md" }, usage: { input: 10, output: 1 } };
	const result = await runV2Mock([
		read, read, read,
		{ type: "final", text: "not-json", usage: { input: 10, output: 1 } },
		{ type: "final", text: JSON.stringify({ ranked_doc_ids: ["doc"] }), usage: { input: 10, output: 1 } },
	]);
	assert.equal(result.valid_output, true);
	assert.equal(result.repair_attempts, 1);
	assert.equal(result.tool_invocations, 3);
});

test("v2 cannot accept a final answer without nonempty read evidence", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const steps: MockStep[] = Array.from({ length: 6 }, () => ({
		type: "tool_call", name: "bash", arguments: { command: "true" }, usage: { input: 10, output: 1 },
	}));
	steps.push({ type: "final", text: JSON.stringify({ ranked_doc_ids: ["doc"] }), usage: { input: 10, output: 1 } });
	steps.push({ type: "final", text: JSON.stringify({ ranked_doc_ids: ["doc"] }), usage: { input: 10, output: 1 } });
	const result = await runV2Mock(steps);
	assert.equal(result.valid_output, false);
	assert.equal(result.failure?.kind, "tool_failure");
	assert.equal(result.repair_attempts, 0);
});

test("v2 reports a cumulative token overrun before final acceptance", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const result = await runV2Mock([
		{ type: "tool_call", name: "read", arguments: { path: "doc.md" }, usage: { input: 64000, output: 1 } },
		{ type: "final", text: JSON.stringify({ ranked_doc_ids: ["doc"] }), usage: { input: 1, output: 1 } },
	]);
	assert.equal(result.valid_output, false);
	assert.equal(result.failure?.kind, "token_limit");
});

test("v2 refuses finalization when one unexpectedly expensive read exhausts the budget", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const result = await runV2Mock([
		{ type: "tool_call", name: "read", arguments: { path: "doc.md" }, usage: { input: 27000, output: 1 } },
		{ type: "final", text: JSON.stringify({ ranked_doc_ids: ["doc"] }), usage: { input: 10, output: 1 } },
	], "32000");
	assert.equal(result.valid_output, false);
	assert.equal(result.failure?.kind, "token_limit");
	assert.equal(result.agent_steps, 1);
	assert.equal(result.repair_attempts, 0);
});
