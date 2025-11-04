#!/usr/bin/env python3
import os, json, subprocess, sys

# --- Step 1: Define paths ---
pcap_path = "capture.pcap"
json_path = "capture.json"

# --- Step 2: Resolve absolute paths ---
pcap_path = os.path.abspath(pcap_path)
json_path = os.path.abspath(json_path)

print(f"🔍 Looking for PCAP at: {pcap_path}")

if not os.path.exists(pcap_path):
    print(f"❌ File not found: {pcap_path}")
    print("💡 Try: python3 01_pcap_to_JSON.py /path/to/file.pcap")
    sys.exit(1)

# --- Step 3: Convert to JSON with tshark ---
cmd = f'tshark -nlr "{pcap_path}" -T json > "{json_path}"'
print(f"🔧 Running: {cmd}")
subprocess.run(cmd, shell=True, check=True)

# --- Step 4: Load and inspect ---
with open(json_path) as f:
    packets = json.load(f)
print(f"\n✅ Parsed {len(packets)} packets from {os.path.basename(pcap_path)}")

if packets:
    print("\n--- First Packet Preview ---")
    print(json.dumps(packets[0], indent=2)[:1500])

# --- Step 5: Clean payloads ---
for pkt in packets:
    layers = pkt.get("_source", {}).get("layers", {})
    for proto in ("tcp", "udp"):
        if proto in layers:
            layers[proto].pop(f"{proto}.payload", None)

with open(json_path, "w") as f:
    json.dump(packets, f, indent=2)

print(f"\n🧹 Cleaned payloads written → {json_path}")
