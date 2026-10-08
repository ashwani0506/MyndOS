"""The conversation loop: persona + profile + history -> brain -> tools -> reply.

Kept separate from the CLI below it because the voice pipeline will drive the
same Agent object later. Run `python agent.py` for a text REPL.
"""

import json
from pathlib import Path
from typing import Callable

import brain
import handlers
import memory
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

    def say(self, text: str, tier: str | None = None) -> str:
        # Before any of this reaches a model. An explicit tier means he asked
        # for one ("/deep what time is it"), so the handlers stand aside --
        # which is also the escape hatch when one of them is wrong.
        #
        # Both sides of the exchange go into history, so a follow-up still has
        # something to refer back to: "and in UTC?" needs the previous answer
        # to exist even though no model produced it.
        if tier is None and (handled := handlers.answer(text)) is not None:
            name, reply = handled
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            # No `ms`: the honest figure is sub-millisecond and the number worth
            # having is the count -- how many turns never touched a model.
            brain._log({"handled": name})
            return reply

        # Routed before the turn goes into history, so the classifier sees the
        # last reply as context rather than the sentence it's classifying.
        # An explicit tier from a caller wins -- that's /fast and /deep.
        if tier is None:
            prev = next(
                (
                    m["content"]
                    for m in reversed(self.history)
                    if m["role"] == "assistant" and m.get("content")
                ),
                "",
            )
            tier = brain.classify(text, prev[:300])

        self.history.append({"role": "user", "content": text})
        # Which EXPLICIT tools this turn is allowed to use, computed from the
        # raw utterance before the model sees it. Recomputed every turn, so a
        # grant never outlives the sentence that asked for it.
        unlocked = tools.unlocked_by(text)
        # Recalled once per turn rather than per hop: the utterance doesn't
        # change mid-turn, and re-reading the notes each hop would only add
        # tokens. Empty string when nothing matches.
        recalled = memory.context(text)

        for _ in range(MAX_HOPS):
            messages = [{"role": "system", "content": system_prompt() + recalled}]
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
                    # The EXPLICIT grant comes from `unlocked`, which was built
                    # from his words alone -- never from the model asking for
                    # it, and never from a file or page it read mid-turn.
                    # CONFIRM tools still stop for a human either way.
                    result = tools.execute(
                        call.function.name,
                        args,
                        user_initiated=call.function.name in unlocked,
                        confirm=self.confirm,
                    )
                self.history.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        return "I got stuck in a tool loop and stopped. Ask me again, more specifically."


def main():
    agent = Agent()
    print("MyndOS (text mode). '/status' for providers, '/tools' for the "
          "registry, '/stats' for routing and latency, '/fast <msg>' or "
          "'/deep <msg>' to override the router, Ctrl-C to quit.\n")
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
        if text == "/stats":
            import measure  # local: the loop doesn't need it until asked

            print(f"\n{measure.report(*measure.load())}\n")
            print(f"{measure.voice_report(*measure.load(measure.VOICE_LOG))}\n")
            continue

        tier = None  # None means let the router pick
        for t in ("fast", "deep"):
            if text.startswith(f"/{t} "):
                tier, text = t, text[len(t) + 2:]

        try:
            print(f"\n{agent.say(text, tier=tier)}\n")
        except brain.NoProviderAvailable as e:
            print(f"\n[no brain] {e}\n")


if __name__ == "__main__":
    main()
