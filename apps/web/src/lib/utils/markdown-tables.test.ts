import { describe, expect, it } from "vitest";
import { mdTable } from "./markdown-tables";

describe("mdTable", () => {
  it("builds a basic table", () => {
    const result = mdTable(
      ["Name", "Value"],
      [
        ["a", "1"],
        ["b", "2"],
      ],
    );
    const lines = result.split("\n");
    expect(lines).toHaveLength(4); // header + separator + 2 rows
    expect(lines[0]).toBe("| Name | Value |");
    expect(lines[1]).toBe("| --- | --- |");
    expect(lines[2]).toBe("| a | 1 |");
    expect(lines[3]).toBe("| b | 2 |");
  });

  it("escapes pipe characters in cells", () => {
    const result = mdTable(["Col"], [["a|b"]]);
    expect(result).toContain("a\\|b");
    expect(result).not.toContain("a|b");
  });

  it("escapes pipe characters in headers", () => {
    const result = mdTable(["X|Y"], [["val"]]);
    expect(result).toContain("X\\|Y");
  });

  it("replaces newlines in cells with spaces", () => {
    const result = mdTable(["Col"], [["line1\nline2"]]);
    expect(result).toContain("line1 line2");
    expect(result).not.toContain("\nline2");
  });

  it("handles empty rows", () => {
    const result = mdTable(["A", "B"], []);
    const lines = result.split("\n");
    expect(lines).toHaveLength(2); // header + separator only
  });

  it("handles single column", () => {
    const result = mdTable(["Only"], [["val"]]);
    expect(result).toContain("| Only |");
    expect(result).toContain("| val |");
  });
});
