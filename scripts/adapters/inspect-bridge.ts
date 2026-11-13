import { createHash } from "node:crypto";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
	createAssistantMessageEventStream,
	fauxToolCall,
	type Api,
	type AssistantMessage,
	type Context,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type Tool,
	type ToolCall,
	type Usage,
} from "@earendil-works/pi-ai";

export type ParsedUsage = { available: boolean; value: Usage };

export type InspectBridgeConfig = {
	baseUrl: string;
	apiKey: string;
	model: string;
	maxTokens: number;
	temperature: number;
	seed: number;
	enableThinking: boolean;
	contextPreflight?: (model: Model<Api>, context: Context) => void;
};

export function emptyUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function numberOrUndefined(value: unknown): number | undefined {
	return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

export function parseUsage(raw: unknown): ParsedUsage {
	if (!raw || typeof raw !== "object") return { available: false, value: emptyUsage() };
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
	if (input === undefined || output === undefined) return { available: false, value: emptyUsage() };
	const hasExplicitTotal = Object.hasOwn(payload, "total_tokens") || Object.hasOwn(payload, "totalTokens");
	const explicitTotal = hasExplicitTotal
		? numberOrUndefined(Object.hasOwn(payload, "total_tokens") ? payload.total_tokens : payload.totalTokens)
		: undefined;
	if (hasExplicitTotal && explicitTotal === undefined) return { available: false, value: emptyUsage() };
	const componentTotal = input + output;
	if (!Number.isSafeInteger(componentTotal) || (explicitTotal !== undefined && explicitTotal < componentTotal)) {
		return { available: false, value: emptyUsage() };
	}
	// Some bridges expose prompt_tokens after cache discounting while total_tokens
	// still reflects the full logical request. Restore that logical input from
	// the total; adding cache or reasoning fields would double count it.
	const logicalInput = explicitTotal === undefined ? input : explicitTotal - output;
	return {
		available: true,
		value: {
			...emptyUsage(),
			input: logicalInput,
			output,
			cacheRead,
			cacheWrite,
			totalTokens: logicalInput + output,
		},
	};
}

export function unavailableUsage(): Usage & { usageAvailable: false } {
	return { ...emptyUsage(), usageAvailable: false };
}

export function availableUsage(value: Usage): Usage & { usageAvailable: true } {
	return { ...value, usageAvailable: true };
}

export function stripModelNoise(text: string): string {
	return text
		.trim()
		.replace(/^(?:(?:<think>[\s\S]*?<\/think>|<<content-internal>[\s\S]*?<\/content-internal>>)\s*)+/, "")
		.trim();
}

export function parseAdaptedJson(text: string): unknown {
	try {
		return JSON.parse(text);
	} catch {
		// Only adapt text outside JSON. Literal thinking tags inside an ID or
		// shell argument belong to the model's payload and must remain intact.
		return JSON.parse(stripModelNoise(text));
	}
}

export function parsePseudoToolCall(text: string): ToolCall | undefined {
	try {
		const value = parseAdaptedJson(text);
		if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
		const parsed = value as Record<string, unknown>;
		if (
			(parsed.name === "read" || parsed.name === "bash") &&
			parsed.arguments &&
			typeof parsed.arguments === "object"
		) {
			return fauxToolCall(parsed.name, parsed.arguments as Record<string, unknown>);
		}
	} catch {
		return undefined;
	}
	return undefined;
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

export function toOpenAIMessages(context: Context): Record<string, unknown>[] {
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

export function toOpenAITools(tools: readonly Tool[] | undefined): Record<string, unknown>[] {
	return (tools ?? []).map((tool) => ({
		type: "function",
		function: {
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters,
		},
	}));
}

type BridgePayload = {
	usage?: unknown;
	choices?: Array<{
		finish_reason?: string;
		message?: {
			content?: unknown;
			tool_calls?: Array<{ id?: string; function?: { name?: string; arguments?: string } }>;
		};
	}>;
};

export function createBridgeStreamFn(config: InspectBridgeConfig): StreamFn {
	return async (model, context, options?: SimpleStreamOptions) => {
		const stream = createAssistantMessageEventStream();
		queueMicrotask(async () => {
			try {
				config.contextPreflight?.(model, context);
				const baseUrl = config.baseUrl.replace(/\/+$/, "");
				const readOnly = context.tools?.length === 1 && context.tools[0].name === "read";
				const request = {
					model: config.model,
					messages: toOpenAIMessages(context),
					tools: toOpenAITools(context.tools),
					tool_choice: options?.toolChoice === "none"
						? "none"
						: readOnly ? { type: "function", function: { name: "read" } } : "auto",
					max_tokens: config.maxTokens,
					temperature: config.temperature,
					seed: config.seed,
					chat_template_kwargs: { enable_thinking: config.enableThinking },
				};
				const response = await fetch(`${baseUrl}/chat/completions`, {
					method: "POST",
					headers: {
						"content-type": "application/json",
						authorization: `Bearer ${config.apiKey}`,
					},
					body: JSON.stringify(request),
					signal: options?.signal,
				});
				const payload = await response.json() as BridgePayload;
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
					const rawArgs = toolCall.function.arguments ?? "{}";
					let parsedArgs: Record<string, unknown> = {};
					try {
						parsedArgs = JSON.parse(rawArgs) as Record<string, unknown>;
					} catch {
						parsedArgs = { raw_arguments: rawArgs };
					}
					content.push({
						type: "toolCall",
						id: toolCall.id ?? `tool_${createHash("sha1").update(JSON.stringify(toolCall)).digest("hex").slice(0, 12)}`,
						name: toolCall.function.name,
						arguments: parsedArgs,
					});
				}
				const pseudoToolCall = !toolsDisabled && content.length === 1 && content[0]?.type === "text"
					? parsePseudoToolCall(content[0].text)
					: undefined;
				if (pseudoToolCall) content.splice(0, content.length, pseudoToolCall);
				const stopReason: AssistantMessage["stopReason"] = choice?.finish_reason === "length"
					? "length"
					: content.some((part) => part.type === "toolCall") ? "toolUse" : "stop";
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
