"""The conversation loop: persona + profile + history -> brain -> tools -> reply.

Kept separate from the CLI below it because the voice pipeline will drive the
same Agent object later. Run `python agent.py` for a text REPL.
"""

import json
from pathlib import Path
from typing import Callable

import brain
import tools

HERE = Path(__file__).parent
MAX_TURNS = 20  # ponytail: flat truncation, replace when the memory layer lands
MAX_HOPS = 5    # tool rounds per message, so a confused model can't loop forever


def system_prompt() -> str:
    """Read persona and profile fresh so edits apply without a restart."""
    persona = (HERE / "persona.md").read_text(encoding="utf-8")
    profile = (HERE / "profile.md").read_text(encoding="utf-8")
    return f"{persona}\n\n---\n\n{profile}"


class Agent:
    def __init__(self, confirm: Callable[..., bool] | None = None):
        self.history: list[dict] = []
        # How CONFIRM-tier tools ask. None means tools.ask -- the terminal
        # prompt. The voice loop passes one that speaks first, so a blocking
        # prompt mid-turn isn't a silent hang.
        self.confirm = confirm

    def say(self, text: str, tier: str = "deep") -> str:
        self.history.append({"role": "user", "content": text})

        for _ in range(MAX_HOPS):
            messages = [{"role": "system", "content": system_prompt()}]
            # Truncation can cut between an assistant tool_calls message and its
            # replies; a leading orphan "tool" message is a 400 from every API.
            window = self.history[-MAX_TURNS:]
            while window and window[0]["role"] == "tool":
                window.pop(0)
            messages += window
            msg = brain.complete(messages, tier=tier, tools=tools.schemas()).choices[0].message
            turn = msg.model_dump(exclude_none=True)
            turn.setdefault("content", None)  # some endpoints require the key present
            self.history.append(turn)

            if not msg.tool_calls:
                return msg.content or ""

            for call in msg.tool_calls:
                # Every tool_call needs exactly one reply appended, or the next
                # request 400s on an assistant message with unanswered calls --
                # so malformed arguments from a small model become a result the
                # model can read and retry, never an exception.
                try:
                    args = json.loads(call.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except ValueError as e:  # JSONDecodeError included
                    result = f"Bad arguments for {call.function.name}: {e}"
                else:
                    # user_initiated stays False: these came from the model,
                    # which may have been reasoning over something it read. The
                    # gate refuses EXPLICIT tools here by design -- I have to
                    # invoke those myself.
                    result = tools.execute(
                        call.function.name, args, confirm=self.confirm
                    )
                self.history.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        return "I got stuck in a tool loop and stopped. Ask me again, more specifically."


def main():
    agent = Agent()
    print("MyndOS (text mode). '/status' for providers, '/tools' for the "
          "registry, '/fast <msg>' for the cheap tier, Ctrl-C to quit.\n")
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
        if text == "/tools":
            for t in tools.REGISTRY.values():
                print(f"  {t.name:<16} [{t.risk.value:<8}] {t.description}")
            print()
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
