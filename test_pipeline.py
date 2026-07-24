import sys
try:
    from src.config import SystemConfig
    from src.temporal_analyzer import TemporalAnalyzer
    from src.state_manager import StateManager
    from src.alarm_controller import AlarmController

    cfg = SystemConfig()
    analyzer = TemporalAnalyzer(cfg)
    state_mgr = StateManager(cfg)
    alarm_ctrl = AlarmController(cfg)

    # Mock 1: Speech
    print("Testing speech mock...")
    raw_ear = 0.3
    raw_mar = 0.6
    for _ in range(10):
        t_state = analyzer.update(raw_ear, raw_mar)
        s_state = state_mgr.update(t_state, True)
        raw_mar = 0.1 if raw_mar == 0.6 else 0.6

    print(f"Speech - is_speaking: {s_state.is_speaking}, yawn_conf: {s_state.yawn_confidence}")

    # Mock 2: Genuine Yawn
    print("\nTesting yawn mock...")
    raw_mar = 0.1
    for i in range(100): # ~3 seconds at 30 fps
        if i > 10: raw_mar = min(0.8, raw_mar + 0.05)
        if i > 90: raw_mar = max(0.1, raw_mar - 0.05)
        
        t_state = analyzer.update(raw_ear, raw_mar)
        s_state = state_mgr.update(t_state, True)
        alarm_ctrl.update(s_state)

    print(f"Yawn ends - total yawns: {s_state.total_yawns}")

    alarm_ctrl.shutdown()
    print("Pipeline integration verified successfully.")

except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
