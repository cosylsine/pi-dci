import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { link, mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { fauxAssistantMessage, type Context, type Model } from "@earendil-works/pi-ai";
import { corpusDocumentIds, createBridgeStreamFn, createReadTool, documentIdIssue, estimateBudgetReserve, parseFinalJson, truncate, utf8Prefix } from "./dci-run.ts";

const PI_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DCI_RUN = join(dirname(fileURLToPath(import.meta.url)), "dci-run.ts");

type MockUsage = { input?: number; output?: number };
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
type RunnerResult = Record<string, unknown> & {
	failure?: { kind?: string } | null;
	stage_timing?: StageTiming;
	trace?: { events?: TraceEvent[]; stage_timing?: StageTiming };
};

async function runV2Mock(steps: MockStep[], tokenBudget?: string): Promise<RunnerResult> {
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
		const child = spawn(process.execPath, ["--import", "tsx", DCI_RUN,
			"--query-file", query, "--workspace", workspace, "--output", output,
			"--trace", trace, "--mock-script", mock, "--model", "mock", "--timeout-seconds", "30"],
			{ cwd: PI_ROOT, env: environment, stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		child.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString("utf8"); });
		child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString("utf8"); });
		const code = await new Promise<number>((resolve, reject) => {
			child.on("error", reject);
			child.on("close", (status) => resolve(status ?? -1));
		});
		const result = JSON.parse(await readFile(output, "utf8")) as RunnerResult;
		result.trace = JSON.parse(await readFile(trace, "utf8")) as RunnerResult["trace"];
		assert.ok(code === 0 || code === 2, `unexpected runner exit ${code}: ${stderr || stdout}`);
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
		const streamFn = createBridgeStreamFn(args);
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

test("v2 dynamic reserve funds only final and one repair from current history", () => {
	const assistant = fauxAssistantMessage("read");
	assistant.usage = { ...assistant.usage, input: 5000, output: 100, totalTokens: 5100 };
	const context: Context = { messages: [
		assistant,
		{ role: "toolResult", toolCallId: "read", toolName: "read", content: [{ type: "text", text: "正文😀" }], isError: false, timestamp: 0 },
	] };
	const reserve = estimateBudgetReserve(context, { reserve_tokens: 0, dynamic_reserve: true });
	assert.ok(reserve.reserve > 0);
	assert.ok(reserve.reserve < 58432);
	assert.ok(reserve.reserve < 64000);
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
	]);
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
	const result = await runV2Mock(steps);
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

test("v2 low calibration budget finalizes or returns a structured token failure", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const result = await runV2Mock([
		{ type: "tool_call", name: "read", arguments: { path: "doc.md" }, usage: { input: 27000, output: 1 } },
		{ type: "final", text: JSON.stringify({ ranked_doc_ids: ["doc"] }), usage: { input: 10, output: 1 } },
	], "32000");
	assert.ok(result.valid_output === true || result.failure?.kind === "token_limit");
});
