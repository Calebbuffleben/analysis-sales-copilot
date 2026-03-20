"""CircularBuffer for raw PCM audio used as the data store for sliding windows.

Conceptually, this buffer stores a fixed-size \"window\" of the most recent
audio samples for a given stream. When new PCM bytes are appended and the
capacity is exceeded, the oldest bytes are discarded automatically.

In our audio pipeline:
- The window size (capacity) is derived from: sample_rate * channels * 2 * window_seconds
  (16-bit PCM = 2 bytes per sample).
- Each stream (meeting_id:participant_id:track) will have its own CircularBuffer.
- Higher-level services (AudioBufferService) will append PCM data for every
  incoming chunk and a SlidingWindowWorker will read the current window
  when it decides that there is enough context to process (transcription / analysis).

This file will later define a CircularBuffer class with operations such as:
- append(data: bytes): add new PCM bytes, dropping the oldest when full.
- read_all() -> bytes: get a copy of the current window (latest PCM samples).
- clear(): reset the buffer.
"""

from __future__ import annotations


class CircularBuffer:
    """Keep the latest PCM bytes up to a fixed capacity."""

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError('capacity_bytes must be greater than zero')

        self.capacity_bytes = capacity_bytes
        self._buffer = bytearray()

    @property
    def current_size_bytes(self) -> int:
        """Return the number of bytes currently stored."""
        return len(self._buffer)

    def append(self, data: bytes) -> None:
        """Append new PCM bytes, discarding the oldest bytes when full."""
        if not data:
            return

        if len(data) >= self.capacity_bytes:
            self._buffer = bytearray(data[-self.capacity_bytes :])
            return

        overflow = (len(self._buffer) + len(data)) - self.capacity_bytes
        if overflow > 0:
            del self._buffer[:overflow]

        self._buffer.extend(data)

    def read_all(self) -> bytes:
        """Return a copy of the current sliding window."""
        return bytes(self._buffer)

    def clear(self) -> None:
        """Reset the buffer contents."""
        self._buffer.clear()

