import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { link, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { createCodeReadTool, parseCodeFiles, parseCodeFinalJson, validateCodeCorpus } from "./dci-run.ts";

function metadata(text: string) {
	return { sha256: createHash("sha256").update(text).digest("hex"),
		line_count: text ? text.split("\n").length - Number(text.endsWith("\n")) : 0 };
}

function sortedJson(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(sortedJson);
	if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => [key, sortedJson(child)]));
	return value;
}

test("code final validates exact paths, abstention, source spans and duplicate JSON keys", () => {
	const files = parseCodeFiles({ "src/report.md": metadata("first\nsecond\n"), "100%/code.py": metadata("x\n") });
	const valid = { ranked_files: ["src/report.md"], abstain: false,
		spans: [{ path: "src/report.md", start_line: 1, end_line: 2 }] };
	assert.deepEqual(parseCodeFinalJson(JSON.stringify(valid), files, 20), valid);
	assert.deepEqual(parseCodeFinalJson('{"ranked_files":[],"abstain":true}', files), { ranked_files: [], abstain: true });
	for (const value of [
		{ ranked_doc_ids: ["src/report.md"] }, { ranked_files: [], abstain: false },
		{ ranked_files: ["src/report.md"], abstain: true }, { ranked_files: ["src/report.md", "src/report.md"], abstain: false },
		{ ranked_files: ["src/../src/report.md"], abstain: false }, { ranked_files: ["/src/report.md"], abstain: false },
		{ ranked_files: ["100%25/code.py"], abstain: false }, { ranked_files: ["src/report"], abstain: false },
		{ ...valid, extra: true }, { ...valid, spans: [{ path: "100%/code.py", start_line: 1, end_line: 1 }] },
		{ ...valid, spans: [{ path: "src/report.md", start_line: 0, end_line: 1 }] },
		{ ...valid, spans: [{ path: "src/report.md", start_line: 1, end_line: 3 }] },
		{ ...valid, spans: [{ path: "src/report.md", start_line: 1, end_line: 2, extra: 1 }] },
	]) assert.equal(parseCodeFinalJson(JSON.stringify(value), files, 20), undefined);
	assert.equal(parseCodeFinalJson('{"ranked_files":[],"abstain":false,"abstain":true}', files), undefined);
	assert.equal(parseCodeFinalJson('{"ranked_files":["src/report.md"],"abstain":false,"spans":[{"path":"src/report.md","start_line":1,"start_line":2,"end_line":2}]}', files), undefined);
	assert.equal(parseCodeFinalJson(JSON.stringify({ ranked_files: Object.keys(files), abstain: false }), files, 1), undefined);
	assert.throws(() => parseCodeFiles({ "../x": metadata("x") }));
	assert.throws(() => parseCodeFiles({ "x": { ...metadata("x"), gold: true } }));
});

test("code corpus preserves UTF-8 hashes and LF lines and rejects extras, links and mismatches", async () => {
	const root = await mkdtemp(join(tmpdir(), "dci-code-manifest-"));
	try {
		const corpus = join(root, "corpus");
		await mkdir(join(corpus, "src"), { recursive: true });
		const text = "\ufeffa\r\nb\u2028c\n";
		await writeFile(join(corpus, "src/x.py"), text);
		const files = parseCodeFiles({ "src/x.py": metadata(text) });
		await validateCodeCorpus(corpus, files);
		await assert.rejects(validateCodeCorpus(corpus, parseCodeFiles({ "src/x.py": metadata("different") })), /mismatch/);
		await writeFile(join(corpus, "extra"), "secret");
		await assert.rejects(validateCodeCorpus(corpus, files), /Unexpected/);
		await rm(join(corpus, "extra"));
		await mkdir(join(corpus, "extra-dir"));
		await assert.rejects(validateCodeCorpus(corpus, files), /Unexpected/);
		await rm(join(corpus, "extra-dir"), { recursive: true });
		await symlink(join(corpus, "src"), join(corpus, "linked"));
		await assert.rejects(validateCodeCorpus(corpus, files), /symlinks/);
		await rm(join(corpus, "linked"));
		await link(join(corpus, "src/x.py"), join(root, "hardlink"));
		await assert.rejects(validateCodeCorpus(corpus, files), /hard links/);
		await rm(join(root, "hardlink"));
		await rm(join(corpus, "src/x.py"));
		await assert.rejects(validateCodeCorpus(corpus, files), /Missing/);
		await writeFile(join(corpus, "src/x.py"), Buffer.from([0xff]));
		await assert.rejects(validateCodeCorpus(corpus, files), /encoded data/);
	} finally { await rm(root, { recursive: true, force: true }); }
});

test("code read reports source ranges, bounds visible UTF-8 and keeps receipts separate from paths", async () => {
	const workspace = await mkdtemp(join(tmpdir(), "dci-code-read-"));
	try {
		const corpus = join(workspace, "corpus");
		await mkdir(join(corpus, "src"), { recursive: true });
		const sources = { "src/a%2Fb.md": "one\n😀second\nthird\n", "empty": "", "blank": "  \n", "trailing-blank": "one\n\n", "large": "中文😀".repeat(8000) };
		for (const [path, text] of Object.entries(sources)) await writeFile(join(corpus, path), text);
		const files = parseCodeFiles(Object.fromEntries(Object.entries(sources).map(([path, text]) => [path, metadata(text)])));
		const evidence = { bodyReadFiles: new Set<string>(), contentSearchCommands: [], readReceipts: [] };
		const tool = createCodeReadTool({ workspace, corpus, bwrap: "/unused", codeFiles: files }, evidence);
		for (const path of ["src/a%2Fb.md", "corpus/src/a%2Fb.md", "/workspace/corpus/src/a%2Fb.md"]) {
			const result = await tool.execute("read", { path, start_line: 2, end_line: 2 });
			assert.deepEqual(result.content, [{ type: "text", text: '{"path":"src/a%2Fb.md","start_line":2,"end_line":2,"last_line_complete":true}\n😀second' }]);
		}
		for (const path of ["../secret", "/etc/passwd", "src/a/b.md", "src/../src/a%2Fb.md"]) {
			await assert.rejects(tool.execute("read", { path }), /Unknown exact/);
		}
		await tool.execute("read", { path: "empty" });
		await tool.execute("read", { path: "blank" });
		const blank = await tool.execute("read", { path: "trailing-blank", start_line: 2, end_line: 2 });
		assert.deepEqual(blank.content, [{ type: "text", text: '{"path":"trailing-blank","start_line":2,"end_line":2,"last_line_complete":true}\n' }]);
		assert.equal(evidence.readReceipts.length, 3);
		await assert.rejects(tool.execute("read", { path: "src/a%2Fb.md", start_line: 4 }), /Invalid source/);
		const result = await tool.execute("read", { path: "large" });
		const output = result.content[0];
		assert.equal(output.type, "text");
		if (output.type === "text") {
			assert.ok(Buffer.byteLength(output.text) <= 20000);
			assert.ok(!output.text.includes("�"));
			assert.ok(output.text.includes('"last_line_complete":false'));
		}
		await writeFile(join(corpus, "large"), "changed");
		await assert.rejects(tool.execute("read", { path: "large" }), /changed/);
		await rm(join(corpus, "src"), { recursive: true });
		await mkdir(join(workspace, "outside"));
		await writeFile(join(workspace, "outside/a%2Fb.md"), sources["src/a%2Fb.md"]);
		await symlink(join(workspace, "outside"), join(corpus, "src"));
		await assert.rejects(tool.execute("read", { path: "src/a%2Fb.md" }), /symlinks/);
	} finally { await rm(workspace, { recursive: true, force: true }); }
});

const piRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
test("code mock runner shares final repair, read gate, usage limits and explicit abstention", {
	skip: !existsSync("/usr/bin/bwrap"),
}, async () => {
	const root = await mkdtemp(join(tmpdir(), "dci-code-online-"));
	try {
		const workspace = join(root, "workspace");
		await mkdir(join(workspace, "corpus/src"), { recursive: true });
		await writeFile(join(workspace, "corpus/src/x.py"), "def useful():\n    return 1\n");
		const files = { "src/x.py": metadata("def useful():\n    return 1\n") };
		await writeFile(join(root, "files.json"), JSON.stringify(files));
		const base = JSON.parse(readFileSync(resolve(piRoot, "../dci_bench/protocol/contract.json"), "utf8"));
		const contract = { ...base, evaluation_track: "code", benchmark_contract_version: "dci-code-online-v1", top_k: 20,
			system_prompt: "Read repository context then return ranked_files, abstain, and optional spans.",
			query_prompt_template: "{query}", repair_prompt_template: "Repair code answer: {reason}" };
		contract.tool_definitions = base.tool_definitions.map((tool: { name: string }) => tool.name === "read" ? {
			...tool, description: "Read source line ranges", input_schema: { type: "object", additionalProperties: false,
				required: ["path"], properties: { path: { type: "string" }, start_line: { type: "integer", minimum: 1 }, end_line: { type: "integer", minimum: 1 } } },
		} : tool);
		await writeFile(join(root, "contract.json"), JSON.stringify(contract));
		const digest = createHash("sha256").update(JSON.stringify(sortedJson(contract))).digest("hex");
		const env = { ...process.env, DCI_EVALUATION_TRACK: "code", DCI_CODE_RUNTIME_CONTRACT: join(root, "contract.json"),
			DCI_PROTOCOL_PROFILE: "deliberately-unknown-document-profile", DCI_PROTOCOL_CONTRACT_SHA256: digest };
		const command = ["--import", "tsx", "scripts/dci-run.ts", "--workspace", workspace,
			"--code-files", join(root, "files.json"), "--query-file", join(root, "query.txt"),
			"--mock-script", join(root, "mock.json"), "--output", join(root, "final.json"), "--trace", join(root, "trace.json")];
		await writeFile(join(root, "query.txt"), "Find useful context");
		const preflight = spawnSync(process.execPath, ["--import", "tsx", "scripts/dci-run.ts", "--workspace", workspace,
			"--code-files", join(root, "files.json"), "--preflight-only"], { cwd: piRoot, env, encoding: "utf8" });
		assert.equal(preflight.status, 0, preflight.stderr);
		const mismatch = spawnSync(process.execPath, command, { cwd: piRoot, env: { ...env, DCI_PROTOCOL_CONTRACT_SHA256: "bad" }, encoding: "utf8" });
		assert.notEqual(mismatch.status, 0);
		assert.match(mismatch.stderr, /contract hash mismatch/);
		for (const abstain of [false, true]) {
			const final = { ranked_files: abstain ? [] : ["src/x.py"], abstain };
			await writeFile(join(root, "mock.json"), JSON.stringify([
				{ type: "final", text: JSON.stringify(final), usage: { input: 10, output: 5 } },
				{ type: "tool_call", name: "read", arguments: { path: "src/x.py", start_line: 1, end_line: 2 }, usage: { input: 10, output: 5 } },
				{ type: "final", text: JSON.stringify(final), usage: { input: 10, output: 5 } },
			]));
			execFileSync(process.execPath, command, { cwd: piRoot, env, encoding: "utf8" });
			const result = JSON.parse(await readFile(join(root, "final.json"), "utf8"));
			assert.equal(result.valid_output, true);
			assert.equal(result.protocol_profile, "code-v1");
			assert.equal(result.base_contract_sha256, digest);
			assert.equal(result.repair_attempts, 1);
			assert.deepEqual(result.code_output, final);
			assert.equal(result.body_evidence.read_receipts.length, 1);
		}
		await writeFile(join(root, "mock.json"), JSON.stringify([
			{ type: "tool_call", name: "read", arguments: { path: "src/x.py" }, usage: { input: 64000, output: 0 } },
			{ type: "final", text: '{"ranked_files":[],"abstain":true}', usage: { input: 64000, output: 0 } },
		]));
		const failed = spawnSync(process.execPath, command, { cwd: piRoot, env, encoding: "utf8" });
		assert.equal(failed.status, 2, failed.stderr);
		const result = JSON.parse(await readFile(join(root, "final.json"), "utf8"));
		assert.equal(result.failure.kind, "token_limit");
		assert.equal(result.code_output, null);
		assert.equal(result.valid_output, false);
	} finally { await rm(root, { recursive: true, force: true }); }
});
