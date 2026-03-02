declare module "jstat" {
  interface Distribution {
    pdf(x: number, ...params: number[]): number;
    cdf(x: number, ...params: number[]): number;
  }

  interface JStatStatic {
    normal: Distribution;
    gamma: Distribution;
    beta: Distribution;
  }

  const jStat: JStatStatic;
  export default jStat;
}
