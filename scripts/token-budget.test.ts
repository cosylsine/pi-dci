import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import type { Context, Message, Tool } from "@earendil-works/pi-ai";
import { forecastTokenBudget } from "./adapters/budget.ts";
import { createTokenCounter } from "./adapters/token-count.ts";

const root = fileURLToPath(new URL("../../", import.meta.url));
const smoke = join(root, "results/Qwen3.8-27B/benchmark-v2-smoke-runs/20261114T081056Z-0mysDb/smoke/evaluation");
const old = join(root, "results/Qwen3.8-27B/benchmark-v2-full-runs/20261113T140604Z-Ap4MVU/dev-96000/evaluation/Qwen3.8-27B/legal_zh/runs/dev-lite/00057/trace.json");
type Event = { action: string; turn: number; remaining_tokens: number };
type Trace = { messages: Message[]; control_events: Event[] };
type Contract = { system_prompt: string; tool_definitions: Array<{ name: string; description: string; input_schema: Tool["parameters"] }>;
	budget_control: { final_prompt: string; final_repair_prompt: string; read_prompt: string } };

async function fixture(path: string) {
	const trace = JSON.parse(await readFile(path, "utf8")) as Trace;
	const manifest = JSON.parse(await readFile(join(dirname(dirname(path)), "run-manifest.json"), "utf8")) as { benchmark_contract: Contract };
	const contract = manifest.benchmark_contract;
	const tools = contract.tool_definitions.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.input_schema }));
	return { trace, contract, context: { systemPrompt: contract.system_prompt, tools, messages: trace.messages } satisfies Context };
}

test("tokenizer forecast counts known contexts and scales only unknown growth", async () => {
	const context: Context = { messages: [{ role: "user", content: "query", timestamp: 0 }] };
	const policy = { final_prompt: "final {reason} {remaining_tokens}", final_repair_prompt: "repair {reason}", read_prompt: "read {remaining_tokens}" };
	const count = async (value: Context) => 1000 + value.messages.length * 10;
	const options = { remaining: 64000, needsRead: false };
	const single = await forecastTokenBudget(context, policy, count, options);
	const array = await forecastTokenBudget(context, policy, count, { ...options, futureToolResults: 3 });
	assert.equal(array.reserve - single.reserve, 2 * 2 * 8192);
	assert.equal(array.finalReserve, single.finalReserve);
	const unread = await forecastTokenBudget(context, policy, count, { ...options, needsRead: true });
	assert.ok(unread.reserve > single.reserve);
	assert.ok(unread.requiredReadReserve! > single.finalReserve);
	assert.ok(single.reserve < 32000);
});

test("recorded smoke requests match tokenizer counts and no longer force early finalization", {
	skip: !process.env.DCI_TEST_LOCAL_TOKENIZER || !existsSync(smoke),
}, async () => {
	const counter = createTokenCounter();
	let checked = 0;
	let continued = 0;
	let minHeadroom = Infinity;
	try {
		const paths = (await readdir(smoke, { recursive: true })).filter((path) => path.endsWith("/trace.json"));
		assert.equal(paths.length, 40);
		for (const relative of paths) {
			const { trace, contract, context } = await fixture(join(smoke, relative));
			let turn = 0;
			for (let index = 0; index < trace.messages.length; index++) {
				const message = trace.messages[index];
				if (message.role !== "assistant") continue;
				const action = trace.control_events.filter((event) => event.turn <= turn).at(-1)?.action;
				const tools = action === "finalize" ? [] : action === "require_read"
					? context.tools.filter((tool) => tool.name === "read") : context.tools;
				const actual = await counter.count({ ...context, tools, messages: trace.messages.slice(0, index) });
				assert.equal(actual, message.usage.input, `${relative} turn ${turn + 1}`);
				checked++;
				turn++;
			}
			const final = trace.control_events.find((event) => event.action === "finalize")!;
			let assistantTurns = 0;
			let boundary = 0;
			for (let index = 0; index < trace.messages.length; index++) {
				if (trace.messages[index].role === "assistant") assistantTurns++;
				if (assistantTurns === final.turn && trace.messages[index].role === "toolResult") boundary = index + 1;
			}
			assert.ok(boundary > 0);
			const forecast = await forecastTokenBudget({ ...context, messages: trace.messages.slice(0, boundary) },
				contract.budget_control, counter.count, { remaining: final.remaining_tokens, needsRead: false });
			assert.ok(forecast.finalReserve < final.remaining_tokens);
			if (forecast.reserve < final.remaining_tokens) continued++;
			minHeadroom = Math.min(minHeadroom, final.remaining_tokens - forecast.reserve);
		}
		assert.equal(continued, 40);
		console.log(JSON.stringify({ smoke_samples: 40, exact_request_counts: checked, allowed_another_search: continued, min_headroom: minHeadroom }));
	} finally { counter.close(); }
});

test("old legal 96k failure reserves a required read earlier and can fund final plus repair", {
	skip: !process.env.DCI_TEST_LOCAL_TOKENIZER || !existsSync(old),
}, async () => {
	const counter = createTokenCounter();
	try {
		const { trace, contract, context } = await fixture(old);
		let spent = 0;
		let turn = 0;
		let firstReadTurn: number | undefined;
		for (let index = 0; index < 16; index++) {
			const message = trace.messages[index];
			if (message.role === "assistant") { spent += message.usage.totalTokens; turn++; }
			if (message.role !== "toolResult") continue;
			const forecast = await forecastTokenBudget({ ...context, messages: trace.messages.slice(0, index + 1) },
				contract.budget_control, counter.count, { remaining: 96000 - spent, needsRead: turn < 7 });
			if (firstReadTurn === undefined && 96000 - spent <= forecast.reserve) firstReadTurn = turn;
			if (turn === 7) {
				assert.ok(forecast.finalReserve < 27982, JSON.stringify(forecast));
				assert.ok(forecast.reserve > 27982);
				console.log(JSON.stringify({ old_legal_first_require_read: firstReadTurn, final_reserve: forecast.finalReserve, remaining: 27982 }));
			}
		}
		assert.ok(firstReadTurn! < 6);
	} finally { counter.close(); }
});

test("exact counter dispatches, finalizes and repairs under 32k/64k/96k; guards block before HTTP", {
	skip: !process.env.DCI_TOKEN_COUNT_PYTHON || !existsSync("/usr/bin/bwrap"),
}, async () => {
	for (const mode of ["32000", "64000", "96000", "token_limit", "context_overflow", "mismatch", "counter_failure"]) {
		const workspace = await mkdtemp(join(tmpdir(), "dci-counted-budget-"));
		let requests = 0;
		let finals = 0;
		const server = createServer(async (request, response) => {
			const chunks: Buffer[] = [];
			for await (const chunk of request) chunks.push(Buffer.from(chunk));
			const body = JSON.parse(Buffer.concat(chunks).toString()) as { messages: Array<{ role: string; content?: string | null }>; tools?: unknown[] };
			const input = mode === "token_limit" ? 31500 : mode === "context_overflow" ? 65000 : 1000 + 100 * body.messages.length;
			response.setHeader("content-type", "application/json");
			if (request.url === "/tokenize") {
				if (mode === "counter_failure" && body.messages.some((message) => message.role === "assistant" &&
					message.content === "<tool_call>invalid final</tool_call>")) {
					response.statusCode = 500;
					response.end("counter unavailable during repair");
					return;
				}
				response.end(JSON.stringify({ count: input }));
				return;
			}
			requests++;
			const final = body.tools?.length === 0;
			if (final) finals++;
			response.end(JSON.stringify({ choices: [{ finish_reason: final ? "stop" : "tool_calls", message: final
				? { content: finals === 1 ? "<tool_call>invalid final</tool_call>" : '{"ranked_doc_ids":["doc"]}' }
				: { content: "", tool_calls: [{ id: `read-${requests}`, type: "function", function: {
					name: "read", arguments: JSON.stringify({ path: "doc.md", offset: requests % 8, limit: 10 }),
				} }] } }], usage: { prompt_tokens: input + (mode === "mismatch" ? 1 : 0), completion_tokens: 10 } }));
		});
		await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
		try {
			await mkdir(join(workspace, "corpus"));
			await writeFile(join(workspace, "corpus/doc.md"), "Relevant document evidence repeated for reads.");
			await writeFile(join(workspace, "query"), "Find relevant documents.");
			const address = server.address();
			assert.ok(address && typeof address !== "string");
			const baseUrl = `http://127.0.0.1:${address.port}/v1`;
			const env: NodeJS.ProcessEnv = { ...process.env, DCI_PROTOCOL_PROFILE: "document-pool-v2-lite",
				DCI_DOCUMENT_TOKEN_BUDGET: mode === "context_overflow" ? "96000" : /^\d+$/.test(mode) ? mode : "32000",
				DCI_TOKEN_COUNT_BASE_URL: baseUrl, DCI_TOKEN_COUNT_MODEL: "scripted", NO_PROXY: "127.0.0.1,localhost", no_proxy: "127.0.0.1,localhost" };
			delete env.DCI_PROTOCOL_CONTRACT_SHA256;
			const child = spawn(process.execPath, ["--import", "tsx", "scripts/dci-run.ts", "--query-file", join(workspace, "query"),
				"--workspace", workspace, "--output", join(workspace, "final.json"), "--trace", join(workspace, "trace.json"),
				"--openai-base-url", baseUrl, "--timeout-seconds", "none"], { cwd: join(root, "pi-dci"), env });
			let stderr = "";
			child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
			child.stdout.resume();
			const code = await new Promise<number | null>((resolve, reject) => { child.on("error", reject); child.on("close", resolve); });
			assert.ok(code === 0 || code === 2, stderr);
			const result = JSON.parse(await readFile(join(workspace, "final.json"), "utf8")) as {
				valid_output: boolean; repair_attempts: number; failure: { kind: string } | null;
				usage_summary: { total_model_tokens: number }; token_checks: Array<{ dispatched: boolean; input_delta?: number }>;
			};
			if (/^\d+$/.test(mode)) {
				assert.equal(result.valid_output, true, JSON.stringify(result));
				assert.equal(result.repair_attempts, 1);
				assert.equal(finals, 2);
				assert.ok(result.usage_summary.total_model_tokens <= Number(mode));
				assert.equal(result.token_checks.length, requests);
				assert.ok(result.token_checks.every((check) => check.dispatched && check.input_delta === 0));
			} else {
				assert.equal(result.valid_output, false);
				assert.equal(result.failure?.kind, mode === "mismatch" || mode === "counter_failure" ? "agent_error" : mode);
				if (mode === "counter_failure") assert.equal(finals, 1);
				else assert.equal(requests, mode === "mismatch" ? 1 : 0);
				assert.equal(result.repair_attempts, 0);
			}
		} finally {
			server.closeAllConnections();
			await new Promise<void>((resolve) => server.close(() => resolve()));
			await rm(workspace, { recursive: true, force: true });
		}
	}
});
