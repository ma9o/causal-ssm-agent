# d10 Case Study — Study Brief

## Study story

A single adult volunteer completed a four-month self-monitoring protocol during a
period of elevated life stress. Once a day they were prompted to report on their
caffeine use, stress, sleep, energy, aches, activity, mood, focus, and social
contact. The aim is idiographic (N-of-1): to reconstruct how this person's daily
states drive one another over time — for example how stress and arousal feed poor
sleep, how sleep and stress shape next-day fatigue, and how fatigue, pain, mood,
focus and social life follow. Measurements are irregular and semantically
heterogeneous (sliders, counts, and instrument readouts), and some days were
missed.

## Constructs

Ten latent constructs are posited. One is **unobserved** (it has no indicator) and
must be inferred from its downstream effects.

- **CaffeineIntake** — the day's caffeine consumption.
- **AutonomicArousal** — background physiological/sympathetic arousal tone. **UNOBSERVED (no indicator).**
- **PerceivedStress** — subjective stress load.
- **SleepQuality** — restfulness of the most recent night's sleep.
- **Fatigue** — daytime tiredness / low energy.
- **MusculoskeletalPain** — bodily aches and pain.
- **PhysicalActivity** — amount of physical movement / exercise.
- **NegativeMood** — negative affect / irritability.
- **CognitiveFocus** — attentional focus and mental sharpness.
- **SocialEngagement** — amount of social contact and interaction.

## Causal structure (DAG)

Directed edges (cause → effect):

```
CaffeineIntake      -> SleepQuality
AutonomicArousal    -> PerceivedStress
AutonomicArousal    -> SleepQuality
AutonomicArousal    -> MusculoskeletalPain
PerceivedStress     -> SleepQuality
PerceivedStress     -> NegativeMood
PerceivedStress     -> Fatigue
SleepQuality        -> Fatigue
Fatigue             -> MusculoskeletalPain
Fatigue             -> PhysicalActivity
Fatigue             -> CognitiveFocus
MusculoskeletalPain -> PhysicalActivity
PhysicalActivity    -> NegativeMood
NegativeMood        -> SocialEngagement
NegativeMood        -> CognitiveFocus
```

A valid topological order:

```
CaffeineIntake, AutonomicArousal, PerceivedStress, SleepQuality, Fatigue,
MusculoskeletalPain, PhysicalActivity, NegativeMood, CognitiveFocus, SocialEngagement
```

## Indicators

Exactly one indicator per observed construct (9 total). `AutonomicArousal` has no
indicator.

| Indicator | Construct | Response type | Family + link |
|---|---|---|---|
| `caffeine_servings` | CaffeineIntake | daily count | poisson + exp |
| `stress_vas` | PerceivedStress | 0–100 slider | gaussian + sigmoid100 |
| `sleep_quality_vas` | SleepQuality | 0–100 slider | gaussian + sigmoid100 |
| `fatigue_score` | Fatigue | continuous | gaussian + identity |
| `pain_nrs` | MusculoskeletalPain | continuous | gaussian + identity |
| `active_minutes` | PhysicalActivity | continuous | gaussian + identity |
| `irritability_index` | NegativeMood | continuous | gaussian + identity |
| `reaction_time_ms` | CognitiveFocus | continuous | gaussian + identity |
| `social_contacts` | SocialEngagement | daily count | poisson + exp |

## Observation design

- **Subject:** one person.
- **Span:** 120 days.
- **Cadence:** one prompt per day, nominally at midday (day index + 0.5).
- **Timing jitter:** each prompt time is perturbed by a uniform ±0.3 day.
- **Missingness:** about 18% of days are missing at random. On every retained day
  all indicators are answered together, so there are no missing cells within a row.
- **File:** `observations.csv` has a `t` column (time in days) followed by one
  column per indicator; count indicators are integers, the rest are floats.

## Ground truth

The data-generating model and all of its parameters live in `hidden/`. The modeler
must not open anything under `hidden/`.
