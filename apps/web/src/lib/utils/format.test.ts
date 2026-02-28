import { describe, expect, it } from "vitest";
import { formatCompact, formatDate, formatDateRange, formatNumber, formatPercent } from "./format";

describe("formatNumber", () => {
  it("formats a positive number with default decimals", () => {
    expect(formatNumber(Math.PI)).toBe("3.142");
  });

  it("formats with custom decimals", () => {
    expect(formatNumber(Math.PI, 1)).toBe("3.1");
  });

  it("returns NaN for NaN", () => {
    expect(formatNumber(Number.NaN)).toBe("NaN");
  });

  it("returns +Inf for positive infinity", () => {
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe("+Inf");
  });

  it("returns -Inf for negative infinity", () => {
    expect(formatNumber(Number.NEGATIVE_INFINITY)).toBe("-Inf");
  });

  it("formats zero", () => {
    expect(formatNumber(0)).toBe("0.000");
  });

  it("formats negative numbers", () => {
    expect(formatNumber(-1.5, 2)).toBe("-1.50");
  });
});

describe("formatPercent", () => {
  it("formats 0.5 as 50%", () => {
    expect(formatPercent(0.5)).toBe("50.0%");
  });

  it("formats 1.0 as 100%", () => {
    expect(formatPercent(1.0)).toBe("100.0%");
  });

  it("formats with custom decimals", () => {
    expect(formatPercent(0.123456, 3)).toBe("12.346%");
  });

  it("formats zero", () => {
    expect(formatPercent(0)).toBe("0.0%");
  });

  it("formats values over 100%", () => {
    expect(formatPercent(1.5)).toBe("150.0%");
  });

  it("formats negative values", () => {
    expect(formatPercent(-0.5)).toBe("-50.0%");
  });

  it("formats NaN", () => {
    expect(formatPercent(Number.NaN)).toBe("NaN%");
  });

  it("formats Infinity", () => {
    expect(formatPercent(Number.POSITIVE_INFINITY)).toBe("Infinity%");
  });
});

describe("formatDate", () => {
  it("formats an ISO date string", () => {
    const result = formatDate("2024-06-15T00:00:00Z");
    expect(result).toContain("Jun");
    expect(result).toContain("15");
    expect(result).toContain("2024");
  });
});

describe("formatDateRange", () => {
  it("formats a date range", () => {
    const result = formatDateRange("2024-01-01T00:00:00Z", "2024-12-31T00:00:00Z");
    expect(result).toContain("Jan");
    expect(result).toContain("Dec");
    expect(result).toContain(" - ");
  });
});

describe("formatCompact", () => {
  it("formats thousands as K", () => {
    expect(formatCompact(1500)).toBe("1.5K");
  });

  it("formats millions as M", () => {
    expect(formatCompact(2_500_000)).toBe("2.5M");
  });

  it("formats small numbers without suffix", () => {
    expect(formatCompact(42)).toBe("42");
  });

  it("formats zero", () => {
    expect(formatCompact(0)).toBe("0");
  });

  it("formats negative numbers", () => {
    const result = formatCompact(-1500);
    expect(result).toContain("1.5");
    expect(result).toContain("K");
    expect(result).toContain("-");
  });
});
