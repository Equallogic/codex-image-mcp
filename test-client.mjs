import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { fileURLToPath } from "node:url";

const transport = new StdioClientTransport({
  command: "node",
  args: [fileURLToPath(new URL("./server.js", import.meta.url))],
  env: { ...process.env },
});
const client = new Client({ name: "test", version: "1.0.0" });
await client.connect(transport);

const { tools } = await client.listTools();
console.log("TOOLS:", tools.map(t => t.name).join(", "));
console.log("SCHEMA KEYS:", Object.keys(tools[0].inputSchema.properties).join(", "));
console.log("REQUIRED:", tools[0].inputSchema.required?.join(", "));

const arg = process.argv[2];
if (arg === "--generate") {
  console.log("\ngenerating...");
  const r = await client.callTool({
    name: "generate_image",
    arguments: {
      prompt: process.argv[3],
      output_path: process.argv[4],
    },
  });
  console.log("isError:", r.isError ?? false);
  console.log(r.content.map(c => c.text).join("\n"));
}
await client.close();
process.exit(0);
