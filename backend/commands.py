import subprocess

async def execute_intent(command: str):
    print(f"[intent] Interpreted Command: {command}")

    if "notepad" in command:
        await run_via_cmd("start notepad")
    elif "calculator" in command:
        await run_via_cmd("start calc")
    elif "paint" in command:
        await run_via_cmd("start mspaint")
    elif "command prompt" in command or "cmd" in command:
        await run_via_cmd("start cmd")
    else:
        print("[intent] No matching app command.")

async def run_via_cmd(cmd: str):
    print(f"[cmd] Running: {cmd}")
    subprocess.Popen(["cmd", "/c", cmd], shell=True)
