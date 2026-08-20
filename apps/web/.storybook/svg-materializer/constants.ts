export const SVG_MATERIALIZER_ENDPOINT = "/__storybook/materialize-svg";
export const SVG_MATERIALIZER_PARAMETER = "svgMaterializer";
export const SVG_MATERIALIZER_TAG = "svg-materialize";

export interface SvgMaterializerParameter {
  /** Exactly one fully laid-out SVG must match this selector. */
  selector: string;
  /** Save once automatically whenever the story is rendered. */
  auto?: boolean;
}

export interface SvgMaterializerResponse {
  relativePath: string;
}
