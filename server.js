#!/usr/bin/env node
/**
 * codex-image-mcp
 *
 * Generates images by driving the Codex CLI bundled inside ChatGPT.app. Codex
 * authenticates from ~/.codex/auth.json, so this runs on the ChatGPT plan and
 * needs no OPENAI_API_KEY. The paid fallback (scripts/image_gen.py) is not used.
 *
 * Three things here are load-bearing:
 *   - The prompt must be imperative and name the save path. Phrased loosely,
 *     Codex replies with the image_gen tool's own prompt-writing guidance
 *     instead of calling it (78s and 22,739 tokens for nothing).
 *   - Suppressing the imagegen skill stops the agent reading its 24KB SKILL.md
 *     and removes a model pass: 80,049 -> 61,197 aggregate input tokens. The
 *     image_gen tool schema itself is only 435 tokens, so the skill doc was
 *     always the real cost, not the tool.
 *   - Codex reports success in prose whether or not a file appeared, so the
 *     only trustworthy signal is the magic bytes on disk.
 *
 * Measure aggregate input, never the "tokens used" line: that figure is
 * cache-discounted, and identical requests were observed receiving between 0
 * and 17,664 cached tokens, so it can move opposite to a config change.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { execFile } from "node:child_process";
import { readFile, stat, mkdir } from "node:fs/promises";
import path from "node:path";
import os from "node:os";

const CODEX_BIN =
  process.env.CODEX_BIN ||
  "/Applications/ChatGPT.app/Contents/Resources/codex";

const CODEX_HOME = process.env.CODEX_HOME || path.join(os.homedir(), ".codex");

// --ignore-user-config drops config.toml (keeping auth), which is the only
// reliable way to strip plugin skills in 0.154.0-alpha.6.2; the per-plugin
// `plugins."x".enabled=false` override is accepted there and silently ignored.
// Because config.toml is gone, the model must be named explicitly.
const AGENT_MODEL = process.env.CODEX_IMAGE_MODEL || "gpt-5.6-sol";

// Suppressing the imagegen skill stops the agent reading its 24KB SKILL.md and
// removes one whole model pass: 80,049 -> 61,197 aggregate input tokens.
const IMAGEGEN_SKILL = path.join(
  CODEX_HOME,
  "skills/.system/imagegen/SKILL.md"
);

const LEAN_FLAGS = [
  "--ephemeral",
  "--ignore-user-config",
  "-m", AGENT_MODEL,
  "-c", 'model_reasoning_effort="low"',
  "-c", 'model_verbosity="low"',
  "-c", "include_apps_instructions=false",
  "-c", "include_environment_context=false",
  "-c", "include_collaboration_mode_instructions=false",
  "-c", `skills.config=[{path="${IMAGEGEN_SKILL}",enabled=false}]`,
  "--json",
];

const TIMEOUT_MS = Number(process.env.CODEX_IMAGE_TIMEOUT_MS || 300_000);

/** Magic bytes, because Codex's prose says "saved" either way. */
const SIGNATURES = [
  { mime: "image/png", bytes: [0x89, 0x50, 0x4e, 0x47] },
  { mime: "image/jpeg", bytes: [0xff, 0xd8, 0xff] },
  { mime: "image/webp", bytes: [0x52, 0x49, 0x46, 0x46] },
];

function sniff(buf) {
  for (const { mime, bytes } of SIGNATURES) {
    if (bytes.every((b, i) => buf[i] === b)) return mime;
  }
  return null;
}

/** PNG dimensions live at a fixed offset in the IHDR chunk. */
function pngSize(buf) {
  if (buf.length < 24 || sniff(buf) !== "image/png") return null;
  return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
}

function runCodex(args, cwd) {
  return new Promise((resolve) => {
    const child = execFile(
      CODEX_BIN,
      args,
      { cwd, timeout: TIMEOUT_MS, maxBuffer: 32 * 1024 * 1024 },
      (error, stdout, stderr) =>
        resolve({ error, stdout: stdout || "", stderr: stderr || "" })
    );
    child.stdin?.end();
  });
}

/**
 * Sum token usage across the JSONL event stream.
 *
 * Codex emits one `turn.completed` event per turn, whose usage is already
 * cumulative over every model pass inside that turn - an image turn is
 * several. Summing across events therefore gives aggregate input, which is the
 * deterministic quantity. The "tokens used" figure Codex prints at the end is
 * cache-discounted instead, and identical requests were measured receiving
 * anywhere from 0 to 17,664 cached tokens, so it can move opposite to a
 * configuration change. Report both, and bill on uncached input + output.
 */
function parseUsage(stdout) {
  const totals = { input: 0, cached: 0, output: 0, turns: 0 };
  for (const line of stdout.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) continue;
    let event;
    try {
      event = JSON.parse(trimmed);
    } catch {
      continue;
    }
    // The usage object moved between builds; accept it wherever it appears.
    const u =
      event.token_usage ||
      event.usage ||
      event.usage_metadata ||
      event.msg?.token_usage ||
      event.msg?.info?.last_token_usage;
    if (!u || typeof u.input_tokens !== "number") continue;
    totals.input += u.input_tokens;
    totals.cached += u.cached_input_tokens || 0;
    // output_tokens already includes reasoning; do not add it again.
    totals.output += u.output_tokens || 0;
    totals.turns += 1;
  }
  return totals.turns ? totals : null;
}

const server = new McpServer({ name: "codex-image", version: "1.0.0" });

server.registerTool(
  "generate_image",
  {
    title: "Generate image",
    description:
      "Generate a raster image with OpenAI's built-in image model via the local " +
      "Codex CLI, and write it to output_path. Runs on the ChatGPT plan, so it " +
      "costs plan quota (roughly 12k billed tokens) rather than API credits. Takes " +
      "roughly 45-60s. Use for photos, illustrations, textures, mockups and " +
      "hero imagery; not for icons, logotype or anything better drawn as SVG. " +
      "Transparent backgrounds are unsupported - ask for a flat, uniform " +
      "background you can key out instead.",
    inputSchema: {
      prompt: z
        .string()
        .min(1)
        .describe(
          "What to depict. Describe subject, composition, lighting, palette " +
            "and materials; vague prompts waste a full 45s generation."
        ),
      output_path: z
        .string()
        .min(1)
        .describe(
          "Where to write the image, ending in .png. Relative paths resolve " +
            "against the current working directory."
        ),
      reference_images: z
        .array(z.string())
        .optional()
        .describe(
          "Optional paths to existing images to use as visual references."
        ),
    },
  },
  async ({ prompt, output_path, reference_images }) => {
    const outPath = path.resolve(output_path.replace(/^~/, os.homedir()));
    const outDir = path.dirname(outPath);
    const outName = path.basename(outPath);

    await mkdir(outDir, { recursive: true });

    // Record mtime so a stale file at this path can't be mistaken for success.
    let priorMtime = null;
    try {
      priorMtime = (await stat(outPath)).mtimeMs;
    } catch {
      /* fresh path */
    }

    const refs = (reference_images || []).map((p) =>
      path.resolve(p.replace(/^~/, os.homedir()))
    );

    // "directly" pairs with the suppressed imagegen skill: it stops the agent
    // going looking for the skill doc it can no longer see. Naming the file
    // also removes any need to parse a path out of Codex's prose.
    let instruction =
      `Call the built-in image generation tool directly. ` +
      `Generate: ${prompt}\n\nSave it as ./${outName}`;
    if (refs.length) {
      instruction +=
        `\n\nUse these existing images as visual references: ` +
        refs.map((r) => `"${r}"`).join(", ");
    }

    const args = [
      "exec",
      "-C", outDir,
      "--sandbox", "workspace-write",
      "--skip-git-repo-check",
      ...LEAN_FLAGS,
      ...refs.flatMap((r) => ["--add-dir", path.dirname(r)]),
      instruction,
    ];

    const started = Date.now();
    const { error, stdout, stderr } = await runCodex(args, outDir);
    const elapsed = ((Date.now() - started) / 1000).toFixed(1);
    const combined = `${stdout}\n${stderr}`;
    const usage = parseUsage(stdout);

    // The file on disk is the only thing worth believing.
    let buf;
    try {
      const info = await stat(outPath);
      if (priorMtime !== null && info.mtimeMs === priorMtime) {
        throw new Error("file was not rewritten");
      }
      buf = await readFile(outPath);
    } catch {
      const tail = combined.trim().split("\n").slice(-12).join("\n");
      return {
        isError: true,
        content: [
          {
            type: "text",
            text:
              `No image was written to ${outPath} after ${elapsed}s.\n` +
              (error ? `Codex exited with: ${error.message}\n` : "") +
              (usage
                ? `Tokens spent anyway: ${usage.input.toLocaleString()} input\n`
                : "") +
              `\nLast output from Codex:\n${tail}`,
          },
        ],
      };
    }

    const mime = sniff(buf);
    if (!mime) {
      return {
        isError: true,
        content: [
          {
            type: "text",
            text:
              `${outPath} exists but is not an image (${buf.length} bytes, no ` +
              `recognised signature). Codex most likely wrote a text or code ` +
              `file instead of calling the image tool.`,
          },
        ],
      };
    }

    const size = pngSize(buf);
    const details = [
      `Wrote ${outPath}`,
      `${mime}${size ? `, ${size.width}x${size.height}` : ""}, ` +
        `${(buf.length / 1024 / 1024).toFixed(2)} MB`,
      usage
        ? `${elapsed}s, ${usage.input.toLocaleString()} input tokens ` +
          `(${usage.cached.toLocaleString()} cached, ` +
          `${(usage.input - usage.cached + usage.output).toLocaleString()} billed)`
        : `${elapsed}s`,
    ].join("\n");

    return { content: [{ type: "text", text: details }] };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
