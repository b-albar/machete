from machete.megakernel.tracing import _max_events_per_lane


def test_trace_event_capacity_handles_wait_heavy_overlap_rows():
    # Qwen backward overlap at S=512 has about 16 tiles per SM and observed
    # controller rows above 600 events. Keep enough headroom to avoid spilling
    # dynamic trace events into adjacent lane rows.
    assert _max_events_per_lane(total_tiles=16 * 70, num_sms=70) >= 1024


def test_trace_event_capacity_scales_with_tiles_per_sm():
    assert _max_events_per_lane(total_tiles=64 * 70, num_sms=70) > _max_events_per_lane(
        total_tiles=16 * 70,
        num_sms=70,
    )
