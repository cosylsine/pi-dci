import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import type { Context } from "@earendil-works/pi-ai";
import { toOpenAIMessages, toOpenAITools } from "./inspect-bridge.ts";

/** Persistent CPU counter. Counts the same normalized payload as Inspect, not JSON bytes. */
export function createTokenCounter() {
	const baseUrl = process.env.DCI_TOKEN_COUNT_BASE_URL;
	const localTokenizer = process.env.DCI_TEST_LOCAL_TOKENIZER;
	const model = process.env.DCI_TOKEN_COUNT_MODEL;
	if ((!baseUrl && !localTokenizer) || !model) {
		throw new Error("V2 requires DCI_TOKEN_COUNT_BASE_URL and DCI_TOKEN_COUNT_MODEL for the vLLM tokenizer");
	}
	const child = spawn(process.env.DCI_TOKEN_COUNT_PYTHON ?? "python3", [
		fileURLToPath(new URL("../../../dci_bench/agents/token_count.py", import.meta.url)),
		...(baseUrl ? ["--base-url", baseUrl] : ["--local-tokenizer", localTokenizer!]), "--model", model,
	], { stdio: ["pipe", "pipe", "pipe"] });
	const pending: Array<{ resolve: (count: number) => void; reject: (error: Error) => void }> = [];
	let failure: Error | undefined;
	let stderr = "";
	child.stderr.on("data", (chunk: Buffer) => { stderr = (stderr + chunk.toString()).slice(-8000); });
	const fail = (error: Error) => {
		failure = error;
		for (const request of pending.splice(0)) request.reject(error);
	};
	child.on("error", fail);
	child.stdin.on("error", fail);
	child.on("close", (code) => fail(new Error(`token counter exited (${code}): ${stderr}`)));
	const lines = createInterface({ input: child.stdout });
	lines.on("line", (line) => {
		const result: unknown = JSON.parse(line);
		if (!result || typeof result !== "object" || !("count" in result) ||
			typeof result.count !== "number" || !Number.isSafeInteger(result.count) || result.count < 0) {
			fail(new Error("Invalid token counter response"));
			return;
		}
		pending.shift()?.resolve(result.count);
	});
	return {
		source: baseUrl ? "vllm-tokenize-inspect" : "offline-tokenizer-inspect",
		count: (context: Context): Promise<number> => {
			if (failure) return Promise.reject(failure);
			return new Promise((resolve, reject) => {
				pending.push({ resolve, reject });
				child.stdin.write(`${JSON.stringify({ messages: toOpenAIMessages(context), tools: toOpenAITools(context.tools),
					chat_template_kwargs: { enable_thinking: false } })}\n`);
			});
		},
		close: () => { lines.close(); child.kill(); },
	};
}
