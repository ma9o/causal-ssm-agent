import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { registerAnalyzeTool } from "./tools/analyze";
import { registerRefineTool } from "./tools/refine";
import { registerResultsTool } from "./tools/results";

const server = new McpServer({
  name: "causal-inference",
  version: "0.1.0",
});

registerAnalyzeTool(server);
registerResultsTool(server);
registerRefineTool(server);

const transport = new StdioServerTransport();
await server.connect(transport);
