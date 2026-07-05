# Domain briefing: one small freshwater lake, summer monitoring season

## The unit and the study

A single small, productive (mesotrophic-to-eutrophic) freshwater lake, observed
at **one fixed monitoring station** near the middle of the lake. During the warm
season the lake is thermally stratified and biologically active. An automated
buoy carrying a multiparameter water-quality sonde logs the physical/chemical
suite, and a technician collects paired grab samples (nutrients, plankton net
tows) on the same visits. Visits are weather-dependent and therefore irregular.

This is an N-of-1 / intensive-longitudinal setting: one lake, followed densely
through time, with irregular and semantically heterogeneous measurements. The
goal is to reason about how the lake's internal state variables drive one
another over the season.

## Constructs

Each construct is a latent state variable of the lake. All but one are tracked
by an indicator; the confounder is unmeasured.

1. **CatchmentLoading** *(latent -- no sensor)*: the intensity of external
   material loading delivered from the surrounding watershed by rainfall and
   runoff. A storm pulse simultaneously washes in dissolved nutrients,
   suspended mineral sediment, and tea-colored terrestrial organic matter, then
   subsides over a few days. There is no direct instrument for "loading"; it is
   inferred only through its several downstream fingerprints.
2. **WaterTemperature**: surface mixed-layer water temperature; the physical
   pace-setter for metabolism and gas solubility.
3. **Nitrate**: dissolved nitrate-nitrogen, the main bioavailable nitrogen pool
   that fuels algal growth.
4. **Turbidity**: optical cloudiness of the water from suspended inorganic
   particles (clay and silt).
5. **CDOM**: colored dissolved organic matter -- the humic, "tea-stained"
   dissolved material leached from the catchment; it tints the water and
   absorbs light.
6. **Phytoplankton**: standing algal biomass in the surface layer.
7. **DissolvedOxygen**: dissolved-oxygen saturation of the surface water.
8. **pH**: acidity/alkalinity of the surface water.
9. **Zooplankton**: abundance of grazing crustacean zooplankton (Daphnia-type
   cladocerans) sampled by vertical net tow.

## Observed constructs, indicators, and measurement metadata

| # | Construct | Indicator (column) | Measurement type | Units | Plausible range |
|---|-----------|--------------------|------------------|-------|-----------------|
| 2 | WaterTemperature | `water_temp_C` | continuous | deg C | ~15-30 |
| 3 | Nitrate | `nitrate_mgL` | continuous | mg/L as N | ~0.05-1.2 |
| 4 | Turbidity | `turbidity_NTU` | continuous | NTU | ~1-40 |
| 5 | CDOM | `fdom_QSU` | continuous | QSU (fluorescence) | ~5-60 |
| 6 | Phytoplankton | `chl_a_ugL` | continuous | ug/L chlorophyll-a | ~2-50 |
| 7 | DissolvedOxygen | `do_sat_pct` | bounded 0-100 index | % saturation | 0-100 |
| 8 | pH | `ph` | continuous | pH units | ~6.5-9.0 |
| 9 | Zooplankton | `zoop_count` | count | individuals per net tow | ~0-200 |

Notes on the indicators:
- `water_temp_C`, `nitrate_mgL`, `turbidity_NTU`, `fdom_QSU`, `chl_a_ugL`, and
  `ph` are read off calibrated continuous instruments / lab assays.
- `do_sat_pct` is reported as percent oxygen saturation and is treated as a
  bounded 0-100 index (this lake's surface water stays below full saturation
  through the sampled season).
- `zoop_count` is an integer count of animals captured in a standardized
  vertical net tow.
- `CatchmentLoading` has **no indicator column** -- it is latent.

## Causal structure (directions only)

Believed directed causal edges, parent -> child. Signs and magnitudes are for
the modeler to infer; only the qualitative shape (proportional vs. saturating)
is annotated where a domain expert would flag it.

- CatchmentLoading -> Nitrate  *(runoff delivers nitrogen)*
- CatchmentLoading -> Turbidity  *(runoff delivers suspended sediment)*
- CatchmentLoading -> CDOM  *(runoff delivers terrestrial organic matter)*
- WaterTemperature -> Nitrate  *(warmth accelerates biological nitrogen uptake/loss)*
- Nitrate -> Phytoplankton  *(nutrient supply for growth)*
- WaterTemperature -> Phytoplankton  *(warmth accelerates growth)*
- Turbidity -> Phytoplankton  *(light limitation; expected saturating)*
- CDOM -> Phytoplankton  *(shading/light limitation)*
- Phytoplankton -> DissolvedOxygen  *(photosynthetic oxygen production)*
- WaterTemperature -> DissolvedOxygen  *(warmer water holds less oxygen)*
- Phytoplankton -> pH  *(CO2 drawdown raises pH)*
- Phytoplankton -> Zooplankton  *(food supply; expected saturating grazing response)*
- WaterTemperature -> Zooplankton  *(warmth accelerates development)*

The two edges marked "expected saturating" are believed to level off at high
parent values (a light-limitation ceiling for phytoplankton, and a
functional-response / satiation ceiling for grazers); the others are expected to
act more proportionally over the observed range.

**Confounding to watch:** `CatchmentLoading` is unmeasured yet drives Nitrate,
Turbidity, and CDOM together. Storm-driven co-movement of those three
indicators therefore reflects a shared hidden common cause, not direct links
among them. Any attempt to attribute phytoplankton response to nitrate alone
must contend with turbidity and CDOM moving in lockstep with it via this hidden
driver.

## Sampling design

- **Span:** ~60 days of a single summer stratification season.
- **Number of observations:** 100 station visits.
- **Cadence:** irregular, weather-dependent. Typical (median) gap between visits
  is roughly half a day; the mean gap is a bit longer because occasional
  weather closures stretch some gaps out to ~2-2.5 days. Visits are sub-daily to
  a couple of days apart, never on a fixed grid.
- All indicators are recorded together at each visit, so every row carries the
  full suite at one timestamp `t` (observation day, a float).

## Qualitative domain context

- **Timescales genuinely differ.** Physical variables such as turbidity and
  dissolved oxygen respond within hours (particles settle fast; gas exchange is
  quick), water temperature turns over the course of a day or so, while
  dissolved organic color, algal biomass, nutrient pools, and especially the
  grazer population carry much longer memory -- the zooplankton respond over a
  generational, multi-day-to-weekly lag. A good model should not force all
  constructs onto one common response time.
- **Nonlinearity is expected in the biology.** Algal biomass self-limits as a
  bloom crowds itself and exhausts local resources, and the grazer population
  runs into a carrying-capacity ceiling; both saturate rather than growing
  without bound. Light limitation of algae and food satiation of grazers are
  likewise expected to be saturating rather than strictly proportional.
- **Loading arrives in pulses.** Watershed inputs are episodic (storm-driven)
  rather than steady, and their signatures in nitrate, turbidity, and color
  decay over a few days as the material settles, dilutes, or is taken up.
- **Chemistry follows biology within a day.** On productive days, photosynthesis
  simultaneously pushes oxygen up and pH up (via CO2 drawdown), while warmer
  water independently lowers oxygen solubility -- so oxygen reflects a tug-of-war
  between biological production and physical solubility.
- **Directions are well accepted; magnitudes are not.** The edge directions
  above reflect standard limnological understanding for a productive lake. The
  strengths, response times, noise levels, and the exact degree of curvature are
  exactly what the modeler is being asked to elicit.
