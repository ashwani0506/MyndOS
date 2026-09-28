"""The conversation loop: persona + profile + history -> brain -> reply.

Kept separate from the CLI below it because the voice pipeline will drive the
same Agent object later. Run `python agent.py` for a text REPL.
"""

from pathlib import Path

import brain

HERE = Path(__file__).parent
MAX_TURNS = 20  # ponytail: flat truncation, replace when the memory layer lands


def system_prompt() -> str:
    """Read persona and profile fresh so edits apply without a restart."""
    persona = (HERE / "persona.md").read_text(encoding="utf-8")
    profile = (HERE / "profile.md").read_text(encoding="utf-8")
    return f"{persona}\n\n---\n\n{profile}"


class Agent:
    def __init__(self):
        self.history: list[dict] = []

    def say(self, text: str, tier: str = "deep") -> str:
        self.history.append({"role": "user", "content": text})
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in self.history[-MAX_TURNS:])
        reply = brain.think(transcript, tier=tier, system=system_prompt())
        self.history.append({"role": "assistant", "content": reply})
        return reply


def main():
    agent = Agent()
    print("MyndOS (text mode). '/status' for providers, '/fast <msg>' for the "
          "cheap tier, Ctrl-C to quit.\n")
    print(brain.status(), "\n")

    while True:
        try:
            text = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if not text:
            continue
        if text == "/status":
            print(brain.status(), "\n")
            continue

        tier = "deep"
        if text.startswith("/fast "):
            tier, text = "fast", text[6:]

        try:
            print(f"\n{agent.say(text, tier=tier)}\n")
        except brain.NoProviderAvailable as e:
            print(f"\n[no brain] {e}\n")


if __name__ == "__main__":
    main()
