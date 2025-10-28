#!/usr/bin/env python3
"""
Packet Copilot Agent
====================
Interactive GPT-4o agent that drives the Packet Copilot FastMCP Server.

1️⃣  Launches the MCP server subprocess
2️⃣  Discovers available tools
3️⃣  Uses GPT-4o to reason and call tools automatically
4️⃣  Streams results back in natural language
"""

import os
import json
import time
import subprocess
import threading
from dotenv import load_dotenv
from openai import OpenAI

# === 1. Environment Setup =====================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# === 2. Launch PacketCopilot MCP Server ======================================
FASTMCP_CMD = ["python3", "server.py"]

mcp_proc = subprocess.Popen(
    FASTMCP_CMD,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=0,
)

def log_stderr(proc):
    for line in proc.stderr:
        print("[MCP STDERR]", line.rstrip())

threading.Thread(target=log_stderr, args=(mcp_proc,), daemon=True).start()

# === 3. JSON-RPC Helpers ======================================================
_request_id = 0
def _next_id():
    global _request_id
    _request_id += 1
    return _request_id

def mcp_send(obj: dict):
    mcp_proc.stdin.write(json.dumps(obj) + "\n")
    mcp_proc.stdin.flush()

def mcp_recv(timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        line = mcp_proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise TimeoutError("No response from MCP server")

# === 4. MCP Lifecycle =========================================================
def initialize_mcp():
    mcp_send({
        "jsonrpc": "2.0",
        "id": _next_id(),
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "packetcopilot-agent", "version": "1.0"},
        },
    })
    time.sleep(0.2)
    mcp_send({"jsonrpc": "2.0", "method": "notifications/initialized"})

def get_tool_list():
    rid = _next_id()
    mcp_send({"jsonrpc": "2.0", "id": rid, "method": "tools/list"})
    while True:
        resp = mcp_recv()
        if resp.get("id") == rid:
            return resp.get("result", {}).get("tools", [])

def call_tool(name: str, args: dict):
    rid = _next_id()
    mcp_send({
        "jsonrpc": "2.0",
        "id": rid,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    })
    while True:
        resp = mcp_recv()
        if resp.get("id") == rid:
            if "error" in resp:
                raise RuntimeError(resp["error"])
            return resp.get("result", {})

def tool_to_openai(tool: dict) -> dict:
    schema = tool.get("inputSchema", {})
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        },
    }

# === 5. GPT + MCP Interactive Agent ==========================================
def react_agent():
    print("[Agent] Initializing Packet Copilot MCP connection …")
    initialize_mcp()
    time.sleep(0.5)

    tools = get_tool_list()
    print(f"[Agent] Discovered tools: {[t['name'] for t in tools]}")
    openai_tools = [tool_to_openai(t) for t in tools]

    messages = [{
        "role": "system",
        "content": (
            "You are Packet Copilot — an expert in packet-level network analysis. "
            "You can use the provided MCP tools to upload PCAPs, convert to JSON, "
            "sanitize, index with embeddings, analyze the capture, and describe it. "
            "When appropriate, call a tool (e.g., 'upload_pcap_base64', 'index_pcap', 'analyze_pcap') "
            "and explain your reasoning clearly to the user."
        ),
    }]

    while True:
        user_input = input("\nAsk something (or 'exit'): ").strip()
        if user_input.lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})
        print(f"[Agent] User → {user_input}")

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments or "{}")
                    print(f"[Agent] Calling tool {name} with {args}")
                    result = call_tool(name, args)

                    messages.append({
                        "role": "assistant",
                        "tool_calls": [call],
                        "content": None,
                    })
                    messages.append({
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result),
                    })

                # Get final natural-language answer
                final = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )
                print("\nAgent →", final.choices[0].message.content)
                messages.append({"role": "assistant", "content": final.choices[0].message.content})

            else:
                print("\nAgent →", msg.content)
                messages.append({"role": "assistant", "content": msg.content})

        except Exception as e:
            err = f"⚠️ Error during tool call or reply: {e}"
            print(err)
            messages.append({"role": "assistant", "content": err})

# === 6. Entrypoint ============================================================
if __name__ == "__main__":
    try:
        react_agent()
    finally:
        if mcp_proc:
            mcp_proc.terminate()
            mcp_proc.wait()
