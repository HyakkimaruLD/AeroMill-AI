"""Sole command entry. Simulator owns slew; stop bypasses the command queue."""

from math import isfinite

from .config import FEED_LIMITS, RPM_LIMITS
from .contracts import (
    AppliedState,
    Command,
    CommandResult,
    ControllerEvent,
    TelemetryChunk,
)


def admissible(rpm: float, feed: float) -> bool:
    return (
        isinstance(rpm, (int, float))
        and isinstance(feed, (int, float))
        and isfinite(rpm)
        and isfinite(feed)
        and RPM_LIMITS[0] <= rpm <= RPM_LIMITS[1]
        and FEED_LIMITS[0] <= feed <= FEED_LIMITS[1]
    )


class Controller:
    def __init__(self, run_id: str, *, fault: str | None = None):
        if fault not in (None, "reject", "lost_ack"):
            raise ValueError("unknown controller fault")
        self.run_id, self.fault = run_id, fault
        self.applied = AppliedState()
        self.pending: Command | None = None
        self.results: dict[tuple[str, str], CommandResult] = {}
        self.action_count = 0

    def apply(self, command: Command) -> CommandResult:
        key = (command.run_id, command.command_id)
        if command.run_id != self.run_id:
            return CommandResult(command.command_id, "ignored", "another run")
        if key in self.results:
            return self.results[key]
        reason = ""
        status = "accepted"
        if command.type == "stop":
            self.applied = AppliedState(self.applied.rpm, 0.0, True)
            self.pending = None
        elif self.applied.stopped:
            reason = "stop latched"
        elif command.type != "set" or not admissible(command.rpm, command.feed_mm_min):
            reason = "invalid command or setpoints outside operating limits"
        elif self.pending is not None:
            reason = "one command already in flight"
        elif self.fault == "reject":
            reason = "injected rejection"
        else:
            self.applied = AppliedState(command.rpm, command.feed_mm_min)
            self.pending = command
            self.action_count += 1
            if self.fault == "lost_ack":
                status = "pending"
        result = CommandResult(
            command.command_id, "rejected" if reason else status, reason
        )
        self.results[key] = result
        return result

    def acknowledge(self, event: ControllerEvent) -> ControllerEvent | None:
        if (
            self.applied.stopped
            or event.run_id != self.run_id
            or self.pending is None
            or event.command_id != self.pending.command_id
        ):
            return None
        return event

    def observe(self, telemetry: TelemetryChunk) -> tuple[ControllerEvent, ...]:
        if (
            self.applied.stopped
            or self.pending is None
            or telemetry.run_id != self.run_id
            or not len(telemetry.rpm)
        ):
            return ()
        if (
            abs(telemetry.rpm[-1] - self.applied.rpm) <= 1
            and abs(telemetry.feed_mm_min[-1] - self.applied.feed_mm_min) <= 1
        ):
            event = ControllerEvent(
                self.pending.command_id,
                "reached",
                float(telemetry.time_s[-1]),
                self.run_id,
            )
            self.pending = None
            return (event,)
        return ()
