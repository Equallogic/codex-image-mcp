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
import { readFile, readdir, realpath, stat, mkdir, mkdtemp, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { inflateSync } from "node:zlib";
import path from "node:path";
import os from "node:os";

// ChatGPT.app moved its bundled Codex in the 2026-10-01 update, from
// Resources/codex to Resources/codex-cli/bin/codex. Take the first candidate
// that exists, so the next move fails loudly instead of with a bare ENOENT.
// An explicit CODEX_BIN is used as given, so a bare name still resolves
// through PATH.
const CODEX_CANDIDATES = [
  "/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex",
  "/Applications/ChatGPT.app/Contents/Resources/codex",
];
const CODEX_BIN =
  process.env.CODEX_BIN ||
  CODEX_CANDIDATES.find((p) => existsSync(p)) ||
  CODEX_CANDIDATES[0];

const CODEX_HOME = process.env.CODEX_HOME || path.join(os.homedir(), ".codex");

// --ignore-user-config drops config.toml (keeping auth), which is the only
// reliable way to strip plugin skills in 0.154.0-alpha.6.2; the per-plugin
// `plugins."x".enabled=false` override is accepted there and silently ignored.
// Because config.toml is gone, the model must be named explicitly.
const AGENT_MODEL = process.env.CODEX_IMAGE_MODEL || "gpt-6.1-sol";

// Second opinions and reviews exist to catch what Claude missed, so they run
// on the strongest model at high effort rather than the cheap image agent.
const REVIEW_MODEL = process.env.CODEX_REVIEW_MODEL || "gpt-6-astra";
const REVIEW_EFFORT = process.env.CODEX_REVIEW_EFFORT || "high";
const REVIEW_TIMEOUT_MS = Number(
  process.env.CODEX_REVIEW_TIMEOUT_MS || 900_000
);

// Skills are multi-step jobs that may render several images, so they get
// more reasoning than the single-shot image agent and a far longer leash.
const SKILL_EFFORT = process.env.CODEX_SKILL_EFFORT || "medium";
const SKILL_TIMEOUT_MS = Number(process.env.CODEX_SKILL_TIMEOUT_MS || 1_800_000);

// Where Codex discovers user skills. Measured under --ignore-user-config:
// these still load, while plugin skills (and computer use with them) do not.
const SKILL_ROOTS = [
  path.join(CODEX_HOME, "skills"),
  path.join(CODEX_HOME, "skills/.system"),
  path.join(os.homedir(), ".agents/skills"),
];

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

/**
 * Fraction of fully transparent pixels in an 8-bit RGBA PNG, or null when the
 * file is some other layout. An alpha channel alone proves nothing: an opaque
 * image saved as RGBA has one too, so the pixels have to be counted.
 */
function transparentFraction(buf) {
  // Bit depth 8, colour type 6 (RGBA), not interlaced.
  if (sniff(buf) !== "image/png" || buf[24] !== 8 || buf[25] !== 6 || buf[28] !== 0) {
    return null;
  }
  const { width, height } = pngSize(buf);
  const idat = [];
  for (let off = 8; off < buf.length; ) {
    const len = buf.readUInt32BE(off);
    if (buf.toString("ascii", off + 4, off + 8) === "IDAT") {
      idat.push(buf.subarray(off + 8, off + 8 + len));
    }
    off += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * 4;
  let prev = Buffer.alloc(stride);
  let clear = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const row = Buffer.from(raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1)));
    for (let i = 0; i < stride; i++) {
      const a = i >= 4 ? row[i - 4] : 0;
      const b = prev[i];
      const c = i >= 4 ? prev[i - 4] : 0;
      let pred = 0;
      if (filter === 1) pred = a;
      else if (filter === 2) pred = b;
      else if (filter === 3) pred = (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      row[i] = (row[i] + pred) & 0xff;
    }
    for (let i = 3; i < stride; i += 4) if (row[i] === 0) clear++;
    prev = row;
  }
  return clear / (width * height);
}

function runCodex(args, cwd, timeout = TIMEOUT_MS) {
  return new Promise((resolve) => {
    const child = execFile(
      CODEX_BIN,
      args,
      { cwd, timeout, maxBuffer: 32 * 1024 * 1024 },
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
  // `codex exec review` emits turn.completed with every count at zero, so a
  // zero total means "not reported", never "free".
  return totals.input ? totals : null;
}

const expand = (p) => path.resolve(p.replace(/^~/, os.homedir()));
const fail = (text) => ({ isError: true, content: [{ type: "text", text }] });

function usageLine(elapsed, usage) {
  return usage
    ? `${elapsed}s, ${usage.input.toLocaleString()} input tokens ` +
        `(${usage.cached.toLocaleString()} cached, ` +
        `${(usage.input - usage.cached + usage.output).toLocaleString()} billed)`
    : `${elapsed}s`;
}

/**
 * Run one image_gen turn and judge it by the file on disk. Generation and
 * editing are the same Codex tool; an edit is a generation whose
 * referenced_image_paths include the image being changed.
 */
async function runImageJob({ instruction, outPath, refs, transparent }) {
  const outDir = path.dirname(outPath);

  await mkdir(outDir, { recursive: true });

  // Record mtime so a stale file at this path can't be mistaken for success.
  let priorMtime = null;
  try {
    priorMtime = (await stat(outPath)).mtimeMs;
  } catch {
    /* fresh path */
  }

  // "directly" pairs with the suppressed imagegen skill: it stops the agent
  // going looking for the skill doc it can no longer see. Naming the file
  // also removes any need to parse a path out of Codex's prose. It has to be
  // absolute: given "./name" on an edit, Codex resolved it against the source
  // image's folder rather than its working directory. The tool's own
  // guidance says to set transparent_background explicitly either way.
  let text =
    `Call the built-in image generation tool directly` +
    (transparent === undefined
      ? `, keeping the source image's transparency exactly as it is. `
      : `, with transparent_background set to ${transparent}. `) +
    `${instruction}\n\nSave it as ${outPath}`;
  if (refs.length) {
    text +=
      `\n\nPass these files as referenced_image_paths: ` +
      refs.map((r) => `"${r}"`).join(", ");
  }

  const args = [
    "exec",
    "-C", outDir,
    "--sandbox", "workspace-write",
    "--skip-git-repo-check",
    ...LEAN_FLAGS,
    ...refs.flatMap((r) => ["--add-dir", path.dirname(r)]),
    text,
  ];

  const started = Date.now();
  const { error, stdout, stderr } = await runCodex(args, outDir);
  const elapsed = ((Date.now() - started) / 1000).toFixed(1);
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
    const tail = `${stdout}\n${stderr}`.trim().split("\n").slice(-12).join("\n");
    return fail(
      `No image was written to ${outPath} after ${elapsed}s.\n` +
        (error ? `Codex exited with: ${error.message}\n` : "") +
        (usage
          ? `Tokens spent anyway: ${usage.input.toLocaleString()} input\n`
          : "") +
        `\nLast output from Codex:\n${tail}`
    );
  }

  const mime = sniff(buf);
  if (!mime) {
    return fail(
      `${outPath} exists but is not an image (${buf.length} bytes, no ` +
        `recognised signature). Codex most likely wrote a text or code ` +
        `file instead of calling the image tool.`
    );
  }

  const size = pngSize(buf);
  const details = [
    `Wrote ${outPath}`,
    `${mime}${size ? `, ${size.width}x${size.height}` : ""}, ` +
      `${(buf.length / 1024 / 1024).toFixed(2)} MB`,
    usageLine(elapsed, usage),
  ];

  // A transparent request is only met if pixels are actually clear.
  if (transparent) {
    const clear = transparentFraction(buf);
    if (!clear) {
      return fail(
        `${details.join("\n")}\n\nTransparency was requested but the file ` +
          (clear === null
            ? `is not an 8-bit RGBA PNG.`
            : `has no fully transparent pixels.`)
      );
    }
    details.push(`${(clear * 100).toFixed(1)}% of pixels fully transparent`);
  }

  return { content: [{ type: "text", text: details.join("\n") }] };
}

/**
 * Run one Codex turn and capture its final message from -o, which is cleaner
 * than fishing the last agent_message out of the JSON stream.
 */
async function runAgent(args, cwd, { model, effort, timeout }) {
  const tmp = await mkdtemp(path.join(os.tmpdir(), "codex-agent-"));
  const lastMessage = path.join(tmp, "last.md");
  try {
    const started = Date.now();
    const { error, stdout, stderr } = await runCodex(
      [
        ...args,
        "--ephemeral",
        "--ignore-user-config",
        "-m", model,
        "-c", `model_reasoning_effort="${effort}"`,
        "--json",
        "-o", lastMessage,
      ],
      cwd,
      timeout
    );
    const elapsed = ((Date.now() - started) / 1000).toFixed(1);
    let answer = "";
    try {
      answer = (await readFile(lastMessage, "utf8")).trim();
    } catch {
      /* no final message */
    }
    const tail = `${stdout}\n${stderr}`.trim().split("\n").slice(-12).join("\n");
    const failText =
      (error ? `Codex exited with: ${error.message}\n` : "") +
      `\nLast output from Codex:\n${tail}`;
    return { ok: !error, answer, elapsed, usage: parseUsage(stdout), failText };
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
}

/**
 * Run a read-only Codex turn and return its final message. Read-only is the
 * point: a second opinion that can change the code it is judging is no
 * longer a second opinion.
 */
async function runReview(args, cwd) {
  const { answer, elapsed, usage, failText } = await runAgent(args, cwd, {
    model: REVIEW_MODEL,
    effort: REVIEW_EFFORT,
    timeout: REVIEW_TIMEOUT_MS,
  });
  if (!answer) {
    return fail(`${REVIEW_MODEL} returned no answer after ${elapsed}s.\n${failText}`);
  }
  return {
    content: [
      {
        type: "text",
        text: `${answer}\n\n---\n${REVIEW_MODEL} (${REVIEW_EFFORT}), ` +
          usageLine(elapsed, usage),
      },
    ],
  };
}

/** Every file under dir with its mtime, so a run can be diffed against it. */
async function snapshot(dir) {
  const files = new Map();
  for (const rel of await readdir(dir, { recursive: true })) {
    const info = await stat(path.join(dir, rel)).catch(() => null);
    if (info?.isFile()) files.set(rel, info.mtimeMs);
  }
  return files;
}

/** One line per written file, judged from its bytes like the image tools. */
async function describeFile(file) {
  const buf = await readFile(file);
  const mime = sniff(buf);
  if (!mime) return `${file} (${buf.length.toLocaleString()} bytes)`;
  const size = pngSize(buf);
  const clear = transparentFraction(buf);
  return (
    `${file} (${mime}${size ? `, ${size.width}x${size.height}` : ""}` +
    (clear ? `, ${(clear * 100).toFixed(1)}% transparent` : "") +
    `)`
  );
}

const server = new McpServer({ name: "codex-image", version: "1.2.0" });

const transparentParam = z
  .boolean()
  .optional()
  .describe(
    "Output a PNG with a transparent background (cutouts, product shots, " +
      "stickers). Verified on disk: fails if no pixel is actually clear."
  );

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
      "To change an existing image, use edit_image instead.",
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
      transparent_background: transparentParam,
    },
  },
  async ({ prompt, output_path, reference_images, transparent_background }) =>
    runImageJob({
      instruction: `Generate: ${prompt}`,
      outPath: expand(output_path),
      refs: (reference_images || []).map(expand),
      transparent: Boolean(transparent_background),
    })
);

server.registerTool(
  "edit_image",
  {
    title: "Edit image",
    description:
      "Change an existing image with OpenAI's image model via the local Codex " +
      "CLI: add or remove elements, recolour, restyle, change the background, " +
      "or cut the subject out onto transparency. Writes a new file and leaves " +
      "the original untouched. Edits are instruction-driven over the whole " +
      "image; there is no mask, so describe the region in words. Same cost " +
      "and time as generate_image.",
    inputSchema: {
      image_path: z.string().min(1).describe("The image to edit."),
      instruction: z
        .string()
        .min(1)
        .describe(
          "What to change, and what must stay the same. Name the region in " +
            "words, e.g. 'replace only the sky with dusk; keep the building " +
            "and framing identical'."
        ),
      output_path: z
        .string()
        .min(1)
        .describe("Where to write the edited image, ending in .png."),
      reference_images: z
        .array(z.string())
        .optional()
        .describe("Optional extra images to borrow style or elements from."),
      transparent_background: transparentParam.describe(
        "true cuts the subject out onto transparency; false fills the " +
          "background in. Omitted keeps whatever the source has. Verified " +
          "on disk: a transparent result fails if no pixel is actually clear."
      ),
    },
  },
  async ({
    image_path,
    instruction,
    output_path,
    reference_images,
    transparent_background,
  }) => {
    const source = expand(image_path);
    const outPath = expand(output_path);
    if (!existsSync(source)) return fail(`No image at ${source}.`);
    // Compare file identity, not strings: macOS is case-insensitive by
    // default and a symlink is another name for the same file.
    const src = await stat(source);
    const dst = await stat(outPath).catch(() => null);
    if (dst && dst.ino === src.ino && dst.dev === src.dev) {
      return fail("output_path is the same file as image_path; edits never overwrite the original.");
    }
    const extras = (reference_images || []).map(expand);
    // Omitted means "keep what the source has": an unrelated edit to a
    // cutout must not quietly fill its background in. A source with clear
    // pixels is held to that on disk; any other source (palette PNG, WebP,
    // partial alpha) gets an instruction to preserve, never one to go opaque.
    const transparent =
      transparent_background ??
      (transparentFraction(await readFile(source)) ? true : undefined);
    return runImageJob({
      instruction:
        `Edit the image "${source}" (the first referenced image). ` +
        `Change: ${instruction}`,
      outPath,
      refs: [source, ...extras],
      transparent,
    });
  }
);

server.registerTool(
  "second_opinion",
  {
    title: "Second opinion from GPT",
    description:
      "Ask OpenAI's strongest model (GPT-6 Astra by default, high effort) for " +
      "an independent answer, via the local Codex CLI on the ChatGPT plan. " +
      "Read-only: it can read files and run read-only commands but cannot " +
      "change anything. Use to check hard math, science or reasoning, to " +
      "challenge a plan or diagnosis, or when stuck. Takes one to several " +
      "minutes. Give it the problem, not your answer, unless you want your " +
      "answer critiqued.",
    inputSchema: {
      question: z
        .string()
        .min(1)
        .describe("The full question with all context it needs; it sees nothing else."),
      files: z
        .array(z.string())
        .optional()
        .describe("Paths it should read before answering."),
      cwd: z
        .string()
        .optional()
        .describe("Directory to work from. Defaults to the server's working directory."),
    },
  },
  async ({ question, files, cwd }) => {
    const dir = cwd ? expand(cwd) : process.cwd();
    let prompt =
      `You are giving an independent second opinion to another AI model, ` +
      `which will weigh your answer against its own. Be direct and specific. ` +
      `Show the decisive reasoning, state your confidence, and say plainly ` +
      `where you disagree with anything in the question.\n\n${question}`;
    if (files?.length) {
      prompt +=
        `\n\nRead these files first:\n` +
        files
          .map((f) => `- ${path.resolve(dir, f.replace(/^~/, os.homedir()))}`)
          .join("\n");
    }
    return runReview(
      ["exec", "-C", dir, "--sandbox", "read-only", "--skip-git-repo-check", prompt],
      dir
    );
  }
);

server.registerTool(
  "review_code",
  {
    title: "Code review by GPT",
    description:
      "Have OpenAI's Codex reviewer (GPT-6 Astra by default) review a git " +
      "change: uncommitted work by default, or a branch against its base, or " +
      "one commit. Read-only. Returns prioritised findings. A different model " +
      "family catches different mistakes, so use it as a check on your own " +
      "work before reporting it done. Takes several minutes.",
    inputSchema: {
      repo_path: z.string().min(1).describe("Path inside the git repository."),
      base: z
        .string()
        .optional()
        .describe("Review the current branch against this base branch."),
      commit: z.string().optional().describe("Review this one commit (SHA)."),
      instructions: z
        .string()
        .optional()
        .describe("Optional focus, e.g. 'concurrency and error handling only'."),
    },
  },
  async ({ repo_path, base, commit, instructions }) => {
    if (base && commit) return fail("Pass base or commit, not both.");
    // The CLI rejects a positional prompt alongside --uncommitted, --base or
    // --commit, so with instructions the target has to be stated in words.
    let args;
    if (!instructions) {
      args = base
        ? ["--base", base]
        : commit
          ? ["--commit", commit]
          : ["--uncommitted"];
    } else {
      const target = base
        ? `the changes on the current branch against base branch "${base}" (git diff ${base}...HEAD)`
        : commit
          ? `the changes introduced by commit ${commit}`
          : `the uncommitted changes: staged, unstaged and untracked`;
      args = [`Review ${target}. Focus: ${instructions}`];
    }
    return runReview(["exec", "review", ...args], expand(repo_path));
  }
);

server.registerTool(
  "run_skill",
  {
    title: "Run a Codex skill",
    description:
      "Run one of the user's Codex skills (from ~/.codex/skills or " +
      "~/.agents/skills) on a task, e.g. black-marlin-packaging-system or " +
      "product-lifestyle-shots. Codex follows the skill with GPT-6.1 Sol and " +
      "its image model, on the ChatGPT plan. It can write only inside " +
      "output_dir, and plugins (computer use, browser) stay off. Returns " +
      "Codex's report plus every file it actually wrote, checked on disk. " +
      "Takes minutes; a multi-image job can take ten or more.",
    inputSchema: {
      skill: z.string().min(1).describe("The skill's folder name."),
      task: z
        .string()
        .min(1)
        .describe("What to make, with every fact the skill needs; it sees nothing else."),
      output_dir: z
        .string()
        .min(1)
        .describe("Folder for everything it produces. Created if missing."),
      input_files: z
        .array(z.string())
        .optional()
        .describe("Source files it should use, e.g. approved product images. Read, never modified."),
    },
  },
  async ({ skill, task, output_dir, input_files }) => {
    if (!SKILL_ROOTS.some((root) => existsSync(path.join(root, skill, "SKILL.md")))) {
      const names = [];
      for (const root of SKILL_ROOTS) {
        const entries = await readdir(root, { withFileTypes: true }).catch(() => []);
        for (const e of entries) {
          if (existsSync(path.join(root, e.name, "SKILL.md"))) names.push(e.name);
        }
      }
      return fail(`No skill named "${skill}". Available: ${names.sort().join(", ")}`);
    }
    const inputs = (input_files || []).map(expand);
    const missing = inputs.filter((f) => !existsSync(f));
    if (missing.length) return fail(`Input files not found: ${missing.join(", ")}`);

    const outDir = expand(output_dir);
    await mkdir(outDir, { recursive: true });
    // The sandbox makes all of output_dir writable, so an input inside it
    // would be protected by nothing but the prompt. Compare real paths so a
    // symlink or case alias can't sneak one in.
    const realOut = await realpath(outDir);
    for (const f of inputs) {
      const rel = path.relative(realOut, await realpath(f));
      if (!rel.startsWith("..") && !path.isAbsolute(rel)) {
        return fail(`Input ${f} is inside output_dir, where Codex could overwrite it. Use a separate output_dir.`);
      }
    }
    const before = await snapshot(outDir);

    let prompt = `Use the $${skill} skill for this task.\n\n${task}`;
    if (inputs.length) {
      prompt += `\n\nInput files:\n` + inputs.map((f) => `- ${f}`).join("\n");
    }
    prompt +=
      `\n\nWrite every file you produce into ${outDir}. ` +
      `Never modify, move or overwrite the input files.`;

    // Inputs are deliberately not --add-dir'd: that would make their folders
    // writable, and the sandbox can already read them.
    const { ok, answer, elapsed, usage, failText } = await runAgent(
      ["exec", "-C", outDir, "--sandbox", "workspace-write", "--skip-git-repo-check", prompt],
      outDir,
      { model: AGENT_MODEL, effort: SKILL_EFFORT, timeout: SKILL_TIMEOUT_MS }
    );

    // As with images, Codex's prose is not evidence; the folder is.
    const after = await snapshot(outDir);
    const written = [...after]
      .filter(([rel, mtime]) => before.get(rel) !== mtime)
      .map(([rel]) => path.join(outDir, rel));
    if (!written.length) {
      return fail(
        `The ${skill} run wrote nothing to ${outDir} after ${elapsed}s.\n` +
          (answer ? `\nCodex said:\n${answer}\n` : "") +
          failText
      );
    }
    const lines = await Promise.all(written.sort().map(describeFile));
    // Files existing is not success: a run that crashed or timed out after
    // writing a scratch prompt must not be reported as a finished job.
    if (!ok) {
      return fail(
        `The ${skill} run failed after ${elapsed}s, leaving partial files:\n` +
          lines.map((l) => `- ${l}`).join("\n") +
          `\n${failText}`
      );
    }
    return {
      content: [
        {
          type: "text",
          text:
            `${answer || "(Codex gave no final message)"}\n\n---\n` +
            `Files written:\n${lines.map((l) => `- ${l}`).join("\n")}\n` +
            `${AGENT_MODEL} (${SKILL_EFFORT}), ${usageLine(elapsed, usage)}`,
        },
      ],
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
