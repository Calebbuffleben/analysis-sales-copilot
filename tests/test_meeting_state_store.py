"""Local fallback store still works without Redis."""

from src.modules.infra.redis_bus import MeetingStateStore


def test_meeting_state_store_local_roundtrip():
    store = MeetingStateStore(redis=None)
    store.set_conversation('t1', 'm1', {'fase_spin': 'problema'})
    store.set_host_context('t1', 'm1', 'host said hello')
    assert store.get_conversation('t1', 'm1')['fase_spin'] == 'problema'
    assert store.get_host_context('t1', 'm1') == 'host said hello'
    assert store.get_conversation('t1', 'missing') == {}
