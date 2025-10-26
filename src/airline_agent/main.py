import asyncio
import os
import traceback
import random
from dotenv import load_dotenv
import json

# preferred import style
from agents import (
    Agent,
    RunContextWrapper,
    handoff,
    function_tool,
    set_tracing_disabled,
)

# ---------------------------
# 1) Disable tracing immediately (call the function)
# ---------------------------
set_tracing_disabled(True)

# ---------------------------
# 2) Load environment
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env (put it in project root).")
print("✅ OpenAI API key loaded successfully.\n")


# ---------------------------
# 3) Context model
# ---------------------------
class AirlineAgentContext:
    """Shared state passed to tools/agents."""
    def __init__(self):
        self.passenger_name: str | None = None
        self.confirmation_number: str | None = None
        self.seat_number: str | None = None
        self.flight_number: str | None = None


# ---------------------------
# 4) Tools (decorated)
# ---------------------------

@function_tool
async def faq_lookup_tool(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ("baggage", "luggage", "carry-on", "bag", "weight")):
        return (
            "You are allowed to bring one bag on the plane. It must be under 50 pounds "
            "and 22 inches x 14 inches x 9 inches."
        )
    if any(k in q for k in ("seat", "seats", "seating", "economy", "business", "exit row")):
        return (
            "There are 120 seats on the plane. There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. Rows 5-8 are Economy Plus, with extra legroom."
        )
    if any(k in q for k in ("wifi", "internet", "connectivity", "network")):
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question"


@function_tool
async def update_seat(context: RunContextWrapper[AirlineAgentContext], confirmation_number: str, new_seat: str) -> str:
    # If flight_number missing, set a generated one (avoid fatal assert here)
    if getattr(context.context, "flight_number", None) is None:
        context.context.flight_number = f"FLT-{random.randint(100,999)}"
    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat
    await asyncio.sleep(0.01)
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number} on flight {context.context.flight_number}"


# ---------------------------
# 5) Robust tool invocation helper
# ---------------------------

async def safe_invoke_tool(tool_obj, *args, **kwargs):
    """
    Robustly invoke a FunctionTool-like object across different agent package versions.
    Attempts several invocation strategies:
      - direct callable (await if coroutine)
      - .run() or .execute()
      - .on_invoke_tool(payload) using params_json_schema or sensible parameter names
      - underlying attributes (.func, .function, __wrapped__)
    Returns the tool result (string, dict, whatever the tool returns).
    """

    # 1) Direct call if callable
    try:
        if callable(tool_obj):
            res = tool_obj(*args, **kwargs)
            if asyncio.iscoroutine(res):
                return await res
            return res
    except Exception:
        # If direct call raises, continue to other attempts (do not bail yet)
        pass

    # 2) .run()
    if hasattr(tool_obj, "run") and callable(getattr(tool_obj, "run")):
        res = tool_obj.run(*args, **kwargs)
        if asyncio.iscoroutine(res):
            return await res
        return res

    # 3) .execute()
    if hasattr(tool_obj, "execute") and callable(getattr(tool_obj, "execute")):
        res = tool_obj.execute(*args, **kwargs)
        if asyncio.iscoroutine(res):
            return await res
        return res

    # 4) on_invoke_tool with params_json_schema
    if hasattr(tool_obj, "on_invoke_tool") and callable(getattr(tool_obj, "on_invoke_tool")):
        on_invoke = getattr(tool_obj, "on_invoke_tool")

        # Attempt to build payload from params_json_schema if available
        schema = getattr(tool_obj, "params_json_schema", None)
        payload = {}

        if schema and isinstance(schema, dict):
            props = schema.get("properties") or {}
            keys = list(props.keys())
            # Map positional args to schema keys where possible (skip RunContextWrapper)
            filtered_args = [a for a in args if not isinstance(a, RunContextWrapper)]
            for i, val in enumerate(filtered_args):
                if i < len(keys):
                    payload[keys[i]] = val
            # Also include kwargs as-is if they match keys
            for k, v in kwargs.items():
                if k in keys:
                    payload[k] = v
        else:
            # no usable schema: attempt heuristics
            # if single positional arg that's a string, try common param names
            if len(args) == 1 and isinstance(args[0], str):
                # common names: question, query, input, text
                payload["question"] = args[0]
                payload["query"] = args[0]
                payload["input"] = args[0]
            else:
                # If we have multiple args and the first is RunContextWrapper, drop it
                filtered_args = [a for a in args if not isinstance(a, RunContextWrapper)]
                # heuristically map to param names
                common_param_names = ["confirmation_number", "new_seat", "seat", "query", "question", "input"]
                for i, val in enumerate(filtered_args):
                    if i < len(common_param_names):
                        payload[common_param_names[i]] = val

        # If a RunContextWrapper is present, pass it separately if the on_invoke_tool supports it.
        run_ctx_arg = None
        for a in args:
            if isinstance(a, RunContextWrapper):
                run_ctx_arg = a
                break

        # Try calling with multiple sensible signatures to maximize compatibility.
        # We'll try positional forms first (since some wrappers accept positional-only args),
        # then payload dict, then kwargs. Also try placing run context first/last.
        filtered_args = [a for a in args if not isinstance(a, RunContextWrapper)]
        positional_attempts = []

        # Provide a minimal dummy context object for wrappers that expect a ToolContext
        class _DummyToolCtx:
            def __init__(self, run_ctx=None):
                # Wrapper expects a .context attribute (RunContextWrapper)
                self.context = run_ctx

        # Primary attempt: pass a ToolContext-like object and a JSON string payload
        positional_attempts.append(lambda: on_invoke(_DummyToolCtx(run_ctx_arg), json.dumps(payload)))

        # Next try: pass the ToolContext-like object and the raw dict payload (some accept a dict)
        positional_attempts.append(lambda: on_invoke(_DummyToolCtx(run_ctx_arg), payload))

        # If the caller provided a single string, try passing it as a single positional arg
        if len(filtered_args) == 1 and isinstance(filtered_args[0], str):
            positional_attempts.append(lambda: on_invoke(filtered_args[0]))

        # Try calling with the original args as-is (may include RunContextWrapper)
        positional_attempts.append(lambda: on_invoke(*args))
        # Try calling with filtered args (drop RunContextWrapper)
        positional_attempts.append(lambda: on_invoke(*filtered_args))

        # Try filtered args with run context appended/prepended
        if run_ctx_arg is not None:
            positional_attempts.append(lambda: on_invoke(*filtered_args, run_ctx_arg))
            positional_attempts.append(lambda: on_invoke(run_ctx_arg, *filtered_args))
            positional_attempts.append(lambda: on_invoke(run_ctx_arg,))

        # Finally try keyword form
        positional_attempts.append(lambda: on_invoke(**payload))

        last_exc = None
        # Print debug info about the wrapper callable
        # optionally introspectable; keep silent in normal operation

        # Add debug-friendly labels for attempts to aid troubleshooting when wrappers vary
        labeled_attempts = []
        for i, fn in enumerate(positional_attempts):
            labeled_attempts.append((f"attempt_{i}", fn))

        for label, attempt_fn in labeled_attempts:
            try:
                # try attempt
                result = attempt_fn()
                if result is None:
                    continue
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as e:
                # record and continue to next attempt
                last_exc = e
                continue

        # If on_invoke_tool attempts all failed, record the last exception and continue
    on_invoke_last_exc = last_exc

    # 5) Underlying wrapped function attributes
    for attr in ("func", "function", "__wrapped__", "_func", "_wrapped"):
        if hasattr(tool_obj, attr):
            underlying = getattr(tool_obj, attr)
            if callable(underlying):
                res = underlying(*args, **kwargs)
                if asyncio.iscoroutine(res):
                    return await res
                return res

    # Nothing worked — list available attributes to help debug
    raise AttributeError(
        "Tool object is not directly callable and no recognized invocation method found. "
        "Available attributes: " + ", ".join(dir(tool_obj))
    )


# ---------------------------
# 6) Agent creation & linking
# ---------------------------

def create_agents():
    # Create instances (no handoffs yet)
    faq_agent = Agent(
        name="FAQ Agent",
        model="gpt-4o-mini",
        instructions="Answer policy questions about baggage, seating, and Wi-Fi.",
    )

    seat_agent = Agent(
        name="Seat Booking Agent",
        model="gpt-4o-mini",
        instructions="Help customers with seat changes and updates. Use update_seat tool.",
    )

    triage_agent = Agent(
        name="Triage Agent",
        model="gpt-4o-mini",
        instructions="Greet the user and route to FAQ or Seat Booking Agent depending on intent.",
    )

    # Attach tools
    faq_agent.tools = [faq_lookup_tool]
    seat_agent.tools = [update_seat]

    # Now wire handoffs using agent instances
    faq_agent.handoffs = [handoff(triage_agent)]
    seat_agent.handoffs = [handoff(triage_agent)]
    triage_agent.handoffs = [handoff(faq_agent), handoff(seat_agent)]

    return triage_agent, faq_agent, seat_agent


# ---------------------------
# 7) Main async flow (robust)
# ---------------------------

async def main():
    print("🚀 Initializing Airline Agents (Tracing Disabled)...\n")

    triage_agent, faq_agent, seat_agent = create_agents()

    # Shared run context object
    shared_ctx = AirlineAgentContext()
    run_ctx = RunContextWrapper(shared_ctx)

    example_queries = [
        "What's the baggage policy?",
        "I want to change my seat",
        "Do you have Wi-Fi?"
    ]

    print("🧭 Simulating agent interactions:\n")

    for q in example_queries:
        try:
            print(f"👤 User: {q}")
            q_low = q.lower()
            if any(k in q_low for k in ("seat", "change my seat", "seat change")):
                # Seat flow — call update_seat using safe_invoke_tool
                resp = await safe_invoke_tool(update_seat, run_ctx, "ABC123", "12A")
                print(f"🤖 Seat Booking Agent: {resp}\n")
            else:
                # FAQ flow
                resp = await safe_invoke_tool(faq_lookup_tool, q)
                print(f"🤖 FAQ Agent: {resp}\n")

        except Exception:
            print("❌ Error during agent handling (stack trace follows):")
            traceback.print_exc()
            print("Continuing to next example...\n")

    print("✅ Simulation complete.\n")


# ---------------------------
# 8) Entrypoint for uv run agent (pyproject)
# ---------------------------

def run_main():
    """Entry point used by pyproject.toml mapping (uv run agent)."""
    asyncio.run(main())


if __name__ == "__main__":
    run_main()
