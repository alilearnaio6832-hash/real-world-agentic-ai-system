# Real-World Agentic AI System

I built this to actually understand how a reliable agent should work — not another LangChain wrapper, but planning, tool use, verification, and recovery built from scratch so I know exactly what's happening at every step.

The core idea driving the design: a stronger LLM should make the system smarter, but a weaker one shouldn't make it wrong. So the LLM only decides *what* to do (which tool, how to phrase things, how to break a task down). A separate, deterministic verification layer decides whether the result is actually correct — completely independent of how confident the LLM sounds.

That's also why verification and recovery got built before planning and the API. Reliability felt more foundational than either.

## How it fits together

```
User Task
   |
   v
run_with_planning()  (optional — breaks the task into ordered steps)
   |
   v
run_with_recovery()
   |
   +--> run_with_tools()   (bounded execution loop)
   |        |
   |        +--> LLM decides: answer, or call a tool?
   |        |       tool call --> Tool Registry --> ToolResult
   |        |       (transient tool failures retried once)
   |        |       empty response --> retried once, then a real error
   |        |
   |        +--> final answer
   |
   +--> Verifier checks the answer
            pass --> done
            fail --> retry, with the failure reason (and full attempt
                     history) fed back into the next try

ExecutionState tracks iterations, tool calls, retries of every kind,
and steps executed — attached to every result.

FastAPI wraps all of this: POST /run, GET /health.
Docker packages the API; it talks to Ollama on the host machine.
```

## What's actually working

- **Two tools**: a safe AST-based calculator (no `eval()`) and a basic text analyzer.
- **Execution loop** with bounded iterations, tool observations fed back into context.
- **Planning**: breaks a task into steps via a constrained prompt format, then runs each step with the previous results as context.
- **Verification**: a rule-based `CalculatorVerifier` that actually parses the LLM's natural-language answer (including LaTeX-style math and multi-step reasoning) and checks it against the real computed answer.
- **Three separate recovery paths**: wrong verification result (retry with feedback), tool execution failure (retry the tool), and empty LLM output (retry the call, then fail loudly instead of pretending an empty string is a valid answer).
- **Evaluation harness** that runs real tasks against a live Ollama model and reports an actual success rate.
- **FastAPI + Docker**, both manually verified end to end.

## What the evaluation actually caught

Running the harness against live Ollama (not mocks) is what makes this project honest. First run: 50% success. Turned out `CalculatorVerifier` was grabbing the first number in the LLM's answer instead of the last one, and choked on parenthesized expressions. Fixed both, re-ran, 100%.

Later, testing the planning flow surfaced something weirder: the model (qwen3.5, which has an internal "thinking" mode) sometimes burns its entire output budget thinking and never actually writes an answer — so you get a response that looks successful but is empty. Found this by inspecting the raw Ollama API response rather than guessing, fixed it by raising the output budget and adding retry logic specifically for empty responses.

Neither of these would have shown up in a mocked unit test. That's the point of the harness.

## Project layout

```
app/
  agent/
    agent.py         run, run_tool, run_with_tools, run_with_verification,
                      run_with_recovery, run_with_planning
    models.py        ToolResult
    state.py         ExecutionState
    planning.py      Planner, Plan
    verification.py  Verifier, VerificationResult, CalculatorVerifier
  llm/               client.py, models.py, ollama.py
  tools/             base.py, calculator.py, text_analyzer.py, registry.py
  evaluation/        evaluator.py, cases.py
  api.py             FastAPI app
scripts/
  run_evaluation.py            calculator benchmark
  run_planning_evaluation.py   planning benchmark
tests/               87+ tests
Dockerfile
.dockerignore
```

## Running it

Needs Python 3.11+ and [Ollama](https://ollama.com) running locally (developed against `qwen3.5:latest`).

```powershell
git clone <repo-url>
cd AI_AGENTIC_SYSTEM
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`:

```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:latest
```

Then:

```powershell
uvicorn app.api:app --reload
```

`http://127.0.0.1:8000/docs` gets you the interactive API.

**Docker:**

```powershell
docker build -t agentic-ai-system .
docker run -p 8000:8000 agentic-ai-system
```

The container reaches Ollama on the host via `host.docker.internal`.

**Tests:** `pytest -v`

**Evaluation:** `python -m scripts.run_evaluation` and `python -m scripts.run_planning_evaluation`

## Roadmap

- [x] V0 — single-tool agent, tool calling, observation
- [x] V0.1 — multi-tool
- [x] V0.2 — planning / task decomposition
- [x] V1 — verification, recovery, execution state, evaluation, API, Docker

Tagged at each milestone: `v0.1.0-multi-tool`, `v0.2.0-planning`, `v1.0.0`.

## License

MIT
