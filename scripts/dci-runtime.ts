import { createHash } from "node:crypto";
import { lstatSync, readFileSync, realpathSync, type Stats } from "node:fs";
import { dirname, isAbsolute, join, sep } from "node:path";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;

export type FrozenRuntimeIdentity = {
	schema_version: "dci-agent-runtime-v1";
	mode: "frozen-rootfs";
	image_digest: `sha256:${string}`;
	platform: "linux/amd64";
	rootfs_tree_sha256: string;
	tool_policy: "open";
	filesystem_policy: "dci-bwrap-frozen-ro-v1";
	runtime_policy_sha256: string;
};

export type FrozenRuntimeDescriptor = {
	identity: FrozenRuntimeIdentity;
	rootfs_path: string;
};

type RuntimeReadyTool = {
	path: "usr/bin/bash" | "usr/bin/rg";
	sha256: string;
	version: string;
};

type RuntimeReady = {
	schema_version: "dci-rootfs-ready-v1";
	content_identity: Pick<FrozenRuntimeIdentity, "image_digest" | "platform" | "rootfs_tree_sha256">;
	tools: { bash: RuntimeReadyTool; rg: RuntimeReadyTool };
};

function isRecord(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function invalidRuntime(message: string): never {
	throw new Error(`Invalid frozen runtime: ${message}`);
}

function exactKeys(value: Record<string, unknown>, expected: readonly string[], label: string): void {
	const actual = Object.keys(value).sort();
	const keys = [...expected].sort();
	if (actual.length !== keys.length || actual.some((key, index) => key !== keys[index])) {
		invalidRuntime(`${label} must contain exactly ${keys.join(", ")}`);
	}
}

function requiredString(value: unknown, label: string): string {
	if (typeof value !== "string" || !value || /[\u0000-\u001f\u007f]/.test(value)) {
		invalidRuntime(`${label} must be a non-empty string`);
	}
	return value;
}

function requiredSha256(value: unknown, label: string): string {
	const text = requiredString(value, label);
	if (!SHA256_PATTERN.test(text)) invalidRuntime(`${label} must be a lowercase SHA-256 hex digest`);
	return text;
}

function requiredImageDigest(value: unknown, label: string): `sha256:${string}` {
	const text = requiredString(value, label);
	if (!/^sha256:[a-f0-9]{64}$/.test(text)) invalidRuntime(`${label} must be sha256:<64 lowercase hex characters>`);
	return text as `sha256:${string}`;
}

function parseJson(text: string, label: string): unknown {
	try {
		return JSON.parse(text) as unknown;
	} catch (error) {
		const reason = error instanceof Error ? error.message : String(error);
		invalidRuntime(`${label} is not valid JSON: ${reason}`);
	}
}

function parseIdentity(value: unknown): FrozenRuntimeIdentity {
	if (!isRecord(value)) invalidRuntime("identity must be an object");
	exactKeys(value, [
		"schema_version", "mode", "image_digest", "platform", "rootfs_tree_sha256", "tool_policy",
		"filesystem_policy", "runtime_policy_sha256",
	], "identity");
	if (value.schema_version !== "dci-agent-runtime-v1") invalidRuntime("identity.schema_version is unsupported");
	if (value.mode !== "frozen-rootfs") invalidRuntime("identity.mode must be frozen-rootfs");
	if (value.platform !== "linux/amd64") invalidRuntime("identity.platform must be linux/amd64");
	if (value.tool_policy !== "open") invalidRuntime("identity.tool_policy must be open");
	if (value.filesystem_policy !== "dci-bwrap-frozen-ro-v1") {
		invalidRuntime("identity.filesystem_policy is unsupported");
	}
	return {
		schema_version: "dci-agent-runtime-v1",
		mode: "frozen-rootfs",
		image_digest: requiredImageDigest(value.image_digest, "identity.image_digest"),
		platform: "linux/amd64",
		rootfs_tree_sha256: requiredSha256(value.rootfs_tree_sha256, "identity.rootfs_tree_sha256"),
		tool_policy: "open",
		filesystem_policy: "dci-bwrap-frozen-ro-v1",
		runtime_policy_sha256: requiredSha256(value.runtime_policy_sha256, "identity.runtime_policy_sha256"),
	};
}

export function parseRuntimeJson(text: string): FrozenRuntimeDescriptor {
	const parsed = parseJson(text, "--runtime-json");
	if (!isRecord(parsed)) invalidRuntime("descriptor must be an object");
	exactKeys(parsed, ["identity", "rootfs_path"], "descriptor");
	const rootfsPath = requiredString(parsed.rootfs_path, "rootfs_path");
	if (!isAbsolute(rootfsPath)) invalidRuntime("rootfs_path must be absolute");
	return { identity: parseIdentity(parsed.identity), rootfs_path: rootfsPath };
}

function statOrThrow(path: string, label: string): Stats {
	try {
		return lstatSync(path);
	} catch (error) {
		const reason = error instanceof Error ? error.message : String(error);
		invalidRuntime(`${label} cannot be inspected: ${reason}`);
	}
}

function realpathOrThrow(path: string, label: string): string {
	try {
		return realpathSync(path);
	} catch (error) {
		const reason = error instanceof Error ? error.message : String(error);
		invalidRuntime(`${label} cannot be resolved: ${reason}`);
	}
}

function assertTrustedReadOnly(info: Stats, label: string): void {
	if ((info.mode & 0o222) !== 0) {
		invalidRuntime(`${label} must be read-only`);
	}
}

function hashFile(path: string, label: string): string {
	try {
		return createHash("sha256").update(readFileSync(path)).digest("hex");
	} catch (error) {
		const reason = error instanceof Error ? error.message : String(error);
		invalidRuntime(`${label} cannot be hashed: ${reason}`);
	}
}

function parseReadyTool(value: unknown, label: string, expectedPath: RuntimeReadyTool["path"]): RuntimeReadyTool {
	if (!isRecord(value)) invalidRuntime(`${label} must be an object`);
	exactKeys(value, ["path", "sha256", "version"], label);
	if (value.path !== expectedPath) invalidRuntime(`${label}.path must be ${expectedPath}`);
	return {
		path: expectedPath,
		sha256: requiredSha256(value.sha256, `${label}.sha256`),
		version: requiredString(value.version, `${label}.version`),
	};
}

function parseReady(value: unknown): RuntimeReady {
	if (!isRecord(value)) invalidRuntime("READY.json must contain an object");
	exactKeys(value, ["schema_version", "content_identity", "tools"], "READY.json");
	if (value.schema_version !== "dci-rootfs-ready-v1") invalidRuntime("READY.json schema_version is unsupported");
	if (!isRecord(value.content_identity)) invalidRuntime("READY.json content_identity must be an object");
	exactKeys(value.content_identity, ["image_digest", "platform", "rootfs_tree_sha256"], "READY.json content_identity");
	if (value.content_identity.platform !== "linux/amd64") {
		invalidRuntime("READY.json content_identity.platform must be linux/amd64");
	}
	if (!isRecord(value.tools)) invalidRuntime("READY.json tools must be an object");
	exactKeys(value.tools, ["bash", "rg"], "READY.json tools");
	return {
		schema_version: "dci-rootfs-ready-v1",
		content_identity: {
			image_digest: requiredImageDigest(value.content_identity.image_digest, "READY.json content_identity.image_digest"),
			platform: "linux/amd64",
			rootfs_tree_sha256: requiredSha256(
				value.content_identity.rootfs_tree_sha256,
				"READY.json content_identity.rootfs_tree_sha256",
			),
		},
		tools: {
			bash: parseReadyTool(value.tools.bash, "READY.json tools.bash", "usr/bin/bash"),
			rg: parseReadyTool(value.tools.rg, "READY.json tools.rg", "usr/bin/rg"),
		},
	};
}

function assertContentIdentity(identity: FrozenRuntimeIdentity, ready: RuntimeReady): void {
	if (ready.content_identity.image_digest !== identity.image_digest ||
		ready.content_identity.platform !== identity.platform ||
		ready.content_identity.rootfs_tree_sha256 !== identity.rootfs_tree_sha256) {
		invalidRuntime("READY.json content_identity does not match identity");
	}
}

function verifyTool(rootfsPath: string, tool: RuntimeReadyTool, label: string): void {
	const path = join(rootfsPath, tool.path);
	const info = statOrThrow(path, label);
	if (info.isSymbolicLink() || !info.isFile()) invalidRuntime(`${label} must be a regular file`);
	const canonical = realpathOrThrow(path, label);
	if (canonical !== rootfsPath && !canonical.startsWith(`${rootfsPath}${sep}`)) {
		invalidRuntime(`${label} resolves outside rootfs`);
	}
	const canonicalInfo = statOrThrow(canonical, label);
	if (!canonicalInfo.isFile() || (canonicalInfo.mode & 0o222) !== 0 || (canonicalInfo.mode & 0o111) === 0) {
		invalidRuntime(`${label} must be a read-only executable`);
	}
	if (hashFile(canonical, label) !== tool.sha256) invalidRuntime(`${label} hash does not match READY.json`);
}

export function validateRuntimeDescriptor(descriptor: FrozenRuntimeDescriptor): FrozenRuntimeDescriptor {
	const rootfsInfo = statOrThrow(descriptor.rootfs_path, "rootfs_path");
	if (rootfsInfo.isSymbolicLink() || !rootfsInfo.isDirectory()) invalidRuntime("rootfs_path must be a real directory");
	assertTrustedReadOnly(rootfsInfo, "rootfs_path");
	const rootfsPath = realpathOrThrow(descriptor.rootfs_path, "rootfs_path");
	const readyPath = join(dirname(rootfsPath), "READY.json");
	const readyInfo = statOrThrow(readyPath, "READY.json");
	if (readyInfo.isSymbolicLink() || !readyInfo.isFile()) invalidRuntime("READY.json must be a regular file");
	assertTrustedReadOnly(readyInfo, "READY.json");
	let ready: RuntimeReady;
	try {
		ready = parseReady(parseJson(readFileSync(readyPath, "utf8"), "READY.json"));
	} catch (error) {
		if (error instanceof Error && error.message.startsWith("Invalid frozen runtime:")) throw error;
		const reason = error instanceof Error ? error.message : String(error);
		invalidRuntime(`READY.json cannot be read: ${reason}`);
	}
	assertContentIdentity(descriptor.identity, ready);
	verifyTool(rootfsPath, ready.tools.bash, "READY.json tools.bash");
	verifyTool(rootfsPath, ready.tools.rg, "READY.json tools.rg");
	return { identity: descriptor.identity, rootfs_path: rootfsPath };
}
