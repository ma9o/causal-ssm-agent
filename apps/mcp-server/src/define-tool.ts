/**
 * Thin wrapper around McpServer.registerTool that avoids TS2589
 * (excessively deep type instantiation) caused by the MCP SDK's
 * complex Zod generic chain.
 *
 * Runtime behavior is identical — Zod validation still happens.
 * We just skip TypeScript's deep inference of callback arg types
 * and use explicit arg interfaces in each tool file instead.
 */
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { z } from "zod";

export type ToolResult = {
  content: Array<{ type: "text"; text: string }>;
  isError?: boolean;
};

export type ToolHandler = (
  args: Record<string, unknown>,
) => Promise<ToolResult>;

export function defineTool(
  server: McpServer,
  name: string,
  description: string,
  inputSchema: z.ZodRawShape,
  handler: ToolHandler,
) {
  // Cast to bypass TS2589: MCP SDK's registerTool() has deeply nested Zod generics
  // that exceed TypeScript's type instantiation depth limit. Runtime behavior is
  // identical — Zod still validates inputs. See: https://github.com/modelcontextprotocol/typescript-sdk/issues/555
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (server as any).registerTool(name, { description, inputSchema }, handler);
}
