/** Escape pipe characters inside a markdown table cell. */
function escapeCell(value: string): string {
  return value.replace(/\|/g, "\\|").replace(/\n/g, " ");
}

/** Build a markdown table from headers and row data. */
export function mdTable(headers: string[], rows: string[][]): string {
  const escaped = rows.map((row) => row.map(escapeCell));
  const headerRow = `| ${headers.map(escapeCell).join(" | ")} |`;
  const separator = `| ${headers.map(() => "---").join(" | ")} |`;
  const bodyRows = escaped.map((row) => `| ${row.join(" | ")} |`);
  return [headerRow, separator, ...bodyRows].join("\n");
}
