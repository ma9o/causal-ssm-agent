"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Pause, Play, RotateCcw } from "lucide-react";

interface AnimationTimelineProps {
  isPlaying: boolean;
  phase: string;
  timeStepsDays: number[];
  currentTimeIndex: number;
  temporalMarkers?: { day: number; label: string }[];
  phaseMarkers?: { position: number; label: string; active: boolean }[];
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onScrub: (timeIndex: number) => void;
}

function findClosestIndex(day: number, days: number[]): number {
  let closest = 0;
  let minDiff = Infinity;
  for (let i = 0; i < days.length; i++) {
    const diff = Math.abs(days[i] - day);
    if (diff < minDiff) {
      minDiff = diff;
      closest = i;
    }
  }
  return closest;
}

export function AnimationTimeline({
  isPlaying,
  phase,
  timeStepsDays,
  currentTimeIndex,
  temporalMarkers,
  phaseMarkers,
  onPlay,
  onPause,
  onReset,
  onScrub,
}: AnimationTimelineProps) {
  const currentDay = timeStepsDays[currentTimeIndex] ?? 0;
  const maxIndex = timeStepsDays.length - 1;

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-3 rounded-lg border bg-card px-4 py-3">
        {/* Transport controls */}
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={isPlaying ? onPause : onPlay}
          >
            {isPlaying ? (
              <Pause className="h-3.5 w-3.5" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
          </Button>
          <Button variant="ghost" size="icon-xs" onClick={onReset}>
            <RotateCcw className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Phase label */}
        <span className="text-xs font-medium text-muted-foreground capitalize min-w-20">
          {phase}
        </span>

        {/* Scrubber */}
        <div className="flex-1 relative">
          <input
            type="range"
            min={0}
            max={maxIndex}
            value={currentTimeIndex}
            onChange={(e) => onScrub(Number(e.target.value))}
            className={cn(
              "w-full h-1.5 rounded-full appearance-none cursor-pointer bg-muted",
              "[&::-webkit-slider-thumb]:appearance-none",
              "[&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5",
              "[&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary",
              "[&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-sm",
            )}
          />

          {/* Tick marks for temporal effect milestones */}
          {temporalMarkers && maxIndex > 0 && (
            <div className="relative h-4 mt-0.5">
              {temporalMarkers.map((marker) => {
                const idx = findClosestIndex(marker.day, timeStepsDays);
                const pct = (idx / maxIndex) * 100;
                return (
                  <div
                    key={marker.label}
                    className="absolute -translate-x-1/2 text-[8px] text-muted-foreground/70"
                    style={{ left: `${pct}%` }}
                  >
                    <div className="w-px h-1.5 bg-muted-foreground/30 mx-auto" />
                    {marker.label}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Current day readout */}
        <span className="text-xs font-mono text-muted-foreground min-w-14 text-right">
          day {currentDay}
        </span>
      </div>

      {/* Rung 3 phase indicators */}
      {phaseMarkers && (
        <div className="relative h-6 px-4">
          {phaseMarkers.map((m) => (
            <span
              key={m.label}
              className={cn(
                "absolute -translate-x-1/2 text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded",
                m.active
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground/50",
              )}
              style={{ left: `${m.position * 100}%` }}
            >
              {m.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
