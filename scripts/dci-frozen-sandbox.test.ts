import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createServer, type AddressInfo } from "node:net";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
	createBashTool,
	DEFAULT_CONTRACT,
	sandboxArgs,
	type CorpusSandbox,
} from "./dci-run.ts";
import { parseRuntimeJson, validateRuntimeDescriptor, type FrozenRuntimeDescriptor } from "./dci-runtime.ts";

const execFileAsync = promisify(execFile);

type BashTool = ReturnType<typeof createBashTool>;
type Fixture = {
	root: string;
	scratch: string;
	bash: BashTool;
	sandbox: CorpusSandbox;
};

function shellQuote(value: string): string {
	return `'${value.replaceAll("'", "'\"'\"'")}'`;
}

function runtimeFromEnvironment(): FrozenRuntimeDescriptor {
	const value = process.env.DCI_TEST_FROZEN_RUNTIME_JSON;
	if (!value) throw new Error("DCI_TEST_FROZEN_RUNTIME_JSON is not set");
	return validateRuntimeDescriptor(parseRuntimeJson(value));
}

async function createFixture(runtime: FrozenRuntimeDescriptor): Promise<Fixture> {
	const root = await mkdtemp(join(tmpdir(), "dci-frozen-sandbox-p5-"));
	const workspace = join(root, "workspace");
	const corpus = join(workspace, "corpus");
	const scratch = join(root, "scratch");
	await mkdir(corpus, { recursive: true });
	await mkdir(join(scratch, "tmp"), { recursive: true, mode: 0o700 });
	await writeFile(join(corpus, "doc.md"), "Frozen sandbox test corpus\n", "utf8");
	const sandbox: CorpusSandbox = { workspace, corpus, bwrap: "/usr/bin/bwrap", runtime };
	const evidence = { bodyReadFiles: new Set<string>(), contentSearchCommands: [], readReceipts: [] };
	return { root, scratch, sandbox, bash: createBashTool(sandbox, scratch, evidence) };
}

async function invoke(
	bash: BashTool,
	command: string,
	timeoutSeconds?: number,
	signal?: AbortSignal,
): Promise<string> {
	const result = await bash.execute("p5-frozen-sandbox", {
		command,
		...(timeoutSeconds === undefined ? {} : { timeout_seconds: timeoutSeconds }),
	}, signal);
	return result.content
		.filter((item): item is { type: "text"; text: string } => item.type === "text")
		.map((item) => item.text)
		.join("\n");
}

async function removeFixture(fixture: Fixture): Promise<void> {
	await rm(fixture.root, { recursive: true, force: true });
}

async function assertNoNamedProcesses(markers: string[]): Promise<void> {
	await new Promise((resolve) => setTimeout(resolve, 250));
	const processList = await execFileAsync("ps", ["-eo", "args="], { encoding: "utf8" });
	const args = String(processList.stdout);
	for (const marker of markers) assert.equal(args.includes(marker), false, `named process remains: ${marker}`);
}

const frozenOnly = { skip: !process.env.DCI_TEST_FROZEN_RUNTIME_JSON, timeout: 30_000 } as const;

test("Frozen scratch is isolated between concurrent calls and persists across calls", frozenOnly, async () => {
	const runtime = runtimeFromEnvironment();
	const first = await createFixture(runtime);
	const second = await createFixture(runtime);
	try {
		const args = sandboxArgs(first.sandbox, first.scratch, "true");
		assert.ok(args.includes("--ro-bind") && args.includes(runtime.rootfs_path));
		assert.ok(args.includes("--unshare-net"));
		assert.equal(args.includes("/usr/bin"), false, "Frozen mode must not bind host /usr/bin");
		const [firstResult, secondResult] = await Promise.all([
			invoke(first.bash, "printf A > /scratch/marker && printf A > /tmp/tmp-marker"),
			invoke(second.bash, "printf B > /scratch/marker && printf B > /tmp/tmp-marker"),
		]);
		assert.match(firstResult, /exit_code=0/);
		assert.match(secondResult, /exit_code=0/);
		assert.equal(await readFile(join(first.scratch, "marker"), "utf8"), "A");
		assert.equal(await readFile(join(second.scratch, "marker"), "utf8"), "B");
		assert.equal(await readFile(join(first.scratch, "tmp", "tmp-marker"), "utf8"), "A");
		assert.equal(await readFile(join(second.scratch, "tmp", "tmp-marker"), "utf8"), "B");
		const firstPersistent = await invoke(first.bash, "test \"$(cat /scratch/marker)\" = A && test \"$(cat /tmp/tmp-marker)\" = A");
		assert.match(firstPersistent, /exit_code=0/);
		const secondPersistent = await invoke(second.bash, "test \"$(cat /scratch/marker)\" = B && test \"$(cat /tmp/tmp-marker)\" = B");
		assert.match(secondPersistent, /exit_code=0/);
		const hiddenOtherScratch = await invoke(first.bash, `test ! -e ${shellQuote(second.scratch + "/marker")}`);
		assert.match(hiddenOtherScratch, /exit_code=0/);
	} finally {
		await Promise.all([removeFixture(first), removeFixture(second)]);
	}
});

test("Frozen /tmp bytes count against the scratch budget", frozenOnly, async () => {
	const runtime = runtimeFromEnvironment();
	const fixture = await createFixture(runtime);
	try {
		const perFile = 800_000;
		const fileCount = Math.ceil((DEFAULT_CONTRACT.max_scratch_bytes + 1) / perFile) + 2;
		const command = [
			"mkdir -p /tmp/dci-p5-budget",
			`for ((i=0; i<${fileCount}; i++)); do printf '%${perFile}s' '' > /tmp/dci-p5-budget/$i; done`,
		].join(" && ");
		const result = await invoke(fixture.bash, command, 60);
		assert.match(result, /\[scratch limit exceeded; process terminated\]/);
		assert.doesNotMatch(result, /\[output limit exceeded; process terminated\]/);
	} finally {
		await removeFixture(fixture);
	}
});

test("Frozen Bash output over the protocol limit terminates the process", frozenOnly, async () => {
	const runtime = runtimeFromEnvironment();
	const fixture = await createFixture(runtime);
	try {
		const result = await invoke(fixture.bash, `printf '%${DEFAULT_CONTRACT.tool_result_max_bytes + 4096}s' ''`);
		assert.match(result, /\[output limit exceeded; process terminated\]/);
	} finally {
		await removeFixture(fixture);
	}
});

test("Frozen Bash timeout and external abort kill named descendants", frozenOnly, async () => {
	const runtime = runtimeFromEnvironment();
	const fixture = await createFixture(runtime);
	try {
		const timeoutMarkers = [
			`dci-p5-timeout-child-${process.pid}`,
			`dci-p5-timeout-grandchild-${process.pid}`,
		];
		const timeoutCommand = [
			"printf started > /scratch/timeout-started;",
			`(exec -a ${timeoutMarkers[0]} /usr/bin/sleep 30) &`,
			`( (exec -a ${timeoutMarkers[1]} /usr/bin/sleep 30) & wait ) &`,
			"wait",
		].join(" ");
		const timeoutResult = await invoke(fixture.bash, timeoutCommand, 1);
		assert.match(timeoutResult, /\[timeout exceeded; process terminated\]/);
		assert.equal(await readFile(join(fixture.scratch, "timeout-started"), "utf8"), "started");
		await assertNoNamedProcesses(timeoutMarkers);

		const abortMarkers = [
			`dci-p5-abort-child-${process.pid}`,
			`dci-p5-abort-grandchild-${process.pid}`,
		];
		const abortCommand = [
			"printf started > /scratch/abort-started;",
			`(exec -a ${abortMarkers[0]} /usr/bin/sleep 30) &`,
			`( (exec -a ${abortMarkers[1]} /usr/bin/sleep 30) & wait ) &`,
			"wait",
		].join(" ");
		const controller = new AbortController();
		const pending = invoke(fixture.bash, abortCommand, 60, controller.signal);
		await new Promise((resolve) => setTimeout(resolve, 150));
		controller.abort();
		const abortResult = await pending;
		assert.match(abortResult, /\[aborted; process terminated\]/);
		assert.equal(await readFile(join(fixture.scratch, "abort-started"), "utf8"), "started");
		await assertNoNamedProcesses(abortMarkers);
	} finally {
		await removeFixture(fixture);
	}
});

test("Frozen sandbox hides host paths and localhost networking", frozenOnly, async () => {
	const runtime = runtimeFromEnvironment();
	const fixture = await createFixture(runtime);
	const hostRoot = await mkdtemp(join(tmpdir(), "dci-p5-host-sentinel-"));
	const hostSentinel = join(hostRoot, "host-only");
	const socketPath = join(hostRoot, "host.sock");
	const unixServer = createServer();
	const tcpServer = createServer();
	try {
		await writeFile(hostSentinel, "host-only\n", "utf8");
		await new Promise<void>((resolve, reject) => {
			unixServer.once("error", reject);
			unixServer.listen(socketPath, () => resolve());
		});
		await new Promise<void>((resolve, reject) => {
			tcpServer.once("error", reject);
			tcpServer.listen(0, "127.0.0.1", () => resolve());
		});
		const address = tcpServer.address();
		assert.ok(address && typeof address === "object");
		const port = (address as AddressInfo).port;
		const command = [
			`if echo probe > /dev/tcp/127.0.0.1/${port} 2>/dev/null; then exit 41; fi`,
			`test ! -e ${shellQuote(hostSentinel)}`,
			`test ! -e ${shellQuote(socketPath)}`,
			"test ! -e /dev/nvidia0 && test ! -e /dev/nvidiactl",
			"test ! -e /proc/1/cmdline",
		].join(" && ");
		const result = await invoke(fixture.bash, command);
		assert.match(result, /exit_code=0/);
	} finally {
		await new Promise<void>((resolve) => unixServer.close(() => resolve()));
		await new Promise<void>((resolve) => tcpServer.close(() => resolve()));
		await rm(socketPath, { force: true });
		await rm(hostRoot, { recursive: true, force: true });
		await removeFixture(fixture);
	}
});
