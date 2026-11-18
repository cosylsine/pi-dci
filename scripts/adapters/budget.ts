import { fauxAssistantMessage, fauxToolCall, type Context } from "@earendil-works/pi-ai";

// Finite feedback vocabulary: no 4000-byte invented ID list in the repair reserve.
export const V2_FINAL_ISSUES = [
	"final answer did not match DCI ranked_doc_ids JSON schema",
	"final answer requires a nonempty corpus read tool receipt",
	"Unknown document IDs. Copy exact doc_id values from corpus results; filenames are not IDs (d000001.md -> d000001).",
] as const;

type Policy = { final_prompt: string; final_repair_prompt: string; read_prompt: string };
type Counter = (context: Context) => Promise<number>;

function user(context: Context, text: string): Context {
	return { ...context, messages: [...context.messages, { role: "user", content: text, timestamp: 0 }] };
}

function futureTool(context: Context, name: string): Context {
	return { ...context, messages: [...context.messages,
		fauxAssistantMessage([fauxToolCall(name, {}, { id: "forecast" })], { stopReason: "toolUse" }),
		{ role: "toolResult", toolCallId: "forecast", toolName: name, content: [{ type: "text", text: "" }],
			isError: false, timestamp: 0 },
	] };
}

/** Known messages/templates are tokenized; only future output and tool text use bounds. */
export async function forecastTokenBudget(context: Context, policy: Policy, count: Counter,
	{ remaining, needsRead, continuationPrompt = "", maxOutput = 1024, toolBytes = 8192,
		futureToolResults = 1 }: { remaining: number; needsRead: boolean; continuationPrompt?: string;
			maxOutput?: number; toolBytes?: number; futureToolResults?: number }) {
	const finalPrompt = policy.final_prompt.replace("{reason}", "remaining budget reserved for final submission")
		.replace("{remaining_tokens}", String(remaining));
	const readPrompt = policy.read_prompt.replace("{remaining_tokens}", String(remaining));
	// Small allowance for unknown message boundaries and future remaining-budget digits.
	// It is not charged on known requests at dispatch.
	const safetyTokens = 32;
	const output = maxOutput;
	const toolGrowth = toolBytes * futureToolResults;
	const growth = output + toolGrowth + safetyTokens;
	const finish = async (base: Context, unknownGrowth: number) => {
		const final = user({ ...base, tools: [] }, finalPrompt);
		const next = { ...final, messages: [...final.messages, fauxAssistantMessage("")] };
		const repairs = await Promise.all(V2_FINAL_ISSUES.map((reason) => count(user(next,
			policy.final_repair_prompt.replace("{reason}", reason)))));
		return await count(final) + unknownGrowth + output + safetyTokens +
			Math.max(...repairs) + unknownGrowth + output * 2 + safetyTokens;
	};
	const requireRead = async (base: Context, unknownGrowth: number) => {
		const read = user({ ...base, tools: context.tools?.filter((tool) => tool.name === "read") }, readPrompt);
		return await count(read) + unknownGrowth + output + safetyTokens +
			await finish(futureTool(read, "read"), unknownGrowth + output + toolBytes + safetyTokens);
	};
	const search = continuationPrompt ? user(context, continuationPrompt) : context;
	const nextInput = await count(search);
	const nextRequest = nextInput + output;
	const finalReserve = await finish(context, 0);
	const requiredReadReserve = needsRead ? await requireRead(context, 0) : undefined;
	const afterSearch = futureTool(search, "bash");
	const reserve = nextRequest + safetyTokens + (needsRead
		? await requireRead(afterSearch, growth) : await finish(afterSearch, growth));
	return { nextInput, nextRequest, finalReserve, requiredReadReserve, reserve, toolGrowth,
		safetyTokens, futureToolResults, count_method: "tokenizer" };
}
