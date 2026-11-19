# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ruiii-newbie/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ruiii-newbie/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z — the percent of tracked players holding it a week after launch — as five quantiles per trophy.

The model works in counts, not just rates: Exophase refreshes its counts in jumps, and when the tracked-player base grows, every rate but the 100% one falls (the newly tracked players hold fewer trophies). It fits the player count and each trophy's earn count as power laws in time since release, shrinks per-trophy slopes toward the cross-trophy median, and — when history is perfectly flat, as on launch night — assumes one refresh before the target, sized like the first refresh the house recorded (player base ~6x, new cohort at ~0.07x the current rate). Quantiles are honest logit-scale intervals around the forecast median.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=575ehchc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
