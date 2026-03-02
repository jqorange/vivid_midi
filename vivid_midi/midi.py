import threading
from collections import deque

import mido


events = deque()
events_lock = threading.Lock()
active_notes: dict[int, int] = {}
active_lock = threading.Lock()


def list_midi_inputs():
    return mido.get_input_names()


def midi_worker(port_name: str):
    with mido.open_input(port_name) as inport:
        for msg in inport:
            if msg.type not in ("note_on", "note_off"):
                continue
            if msg.type == "note_on" and msg.velocity == 0:
                t, vel = "note_off", 0
            else:
                t, vel = msg.type, getattr(msg, "velocity", 0)
            with events_lock:
                events.append((t, msg.note, vel))


def choose_midi_port():
    ports = list_midi_inputs()
    if not ports:
        raise RuntimeError("No MIDI input ports found.")
    print("\nAvailable MIDI input ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")
    while True:
        s = input("\nSelect MIDI input by number: ").strip()
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(ports):
                return ports[idx]
        print("Invalid selection.")


def drain_events():
    with events_lock:
        local = list(events)
        events.clear()
    return local


def apply_events(local_events, note_min: int, note_max: int):
    if not local_events:
        return
    with active_lock:
        for t, note, vel in local_events:
            if note < note_min or note > note_max:
                continue
            if t == "note_on":
                active_notes[note] = int(vel)
            elif t == "note_off":
                active_notes.pop(note, None)


def get_held_notes():
    with active_lock:
        return list(active_notes.items())
