"""Temporal buffer for remote seller fingerprints."""

from __future__ import annotations

from .types import AudioFingerprint


class FingerprintBuffer:
    """Ring buffer keyed by seller user id."""

    def __init__(self, ttl_ms: int = 5000) -> None:
        self._ttl_ms = ttl_ms
        self._entries: dict[str, list[AudioFingerprint]] = {}

    def add(self, fingerprint: AudioFingerprint) -> None:
        bucket = self._entries.setdefault(fingerprint.user_id, [])
        bucket.append(fingerprint)
        self._prune(fingerprint.user_id, now_ms=fingerprint.capture_time_ms)

    def add_many(self, fingerprints: list[AudioFingerprint]) -> None:
        for fp in fingerprints:
            self.add(fp)

    def prune(self, now_ms: int) -> None:
        for user_id in list(self._entries):
            self._prune(user_id, now_ms=now_ms)

    def _prune(self, user_id: str, *, now_ms: int) -> None:
        bucket = self._entries.get(user_id)
        if not bucket:
            return
        cutoff = now_ms - self._ttl_ms
        self._entries[user_id] = [fp for fp in bucket if fp.capture_time_ms >= cutoff]
        if not self._entries[user_id]:
            del self._entries[user_id]

    def candidates(
        self,
        user_id: str,
        *,
        center_ms: int,
        max_lag_ms: int,
    ) -> list[AudioFingerprint]:
        bucket = self._entries.get(user_id, [])
        low = center_ms - max_lag_ms
        high = center_ms + max_lag_ms
        return [fp for fp in bucket if low <= fp.capture_time_ms <= high]

    def all_user_ids(self) -> list[str]:
        return list(self._entries.keys())

    def snapshot(self) -> dict[str, list[AudioFingerprint]]:
        return {uid: list(fps) for uid, fps in self._entries.items()}
