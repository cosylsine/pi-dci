import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { link, mkdtemp, mkdir, rm, symlink, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fauxAssistantMessage, type Context, type Model } from "@earendil-works/pi-ai";
import { corpusDocumentIds, createBridgeStreamFn, createReadTool, documentIdIssue, estimateBudgetReserve, parseFinalJson, truncate, utf8Prefix } from "./dci-run.ts";

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
