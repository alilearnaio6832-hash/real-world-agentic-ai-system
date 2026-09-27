\[9/27/2026 2:50 PM] Myself IR\_MCI: ## به‌روزرسانی README.md



notepad README.md

فایل رو کامل با محتوای زیر جایگزین کن:



\# Real-World Agentic AI System



A measurable, reliable, modular agentic AI system built from first

principles — not a LangChain wrapper. This project demonstrates real

agentic engineering: planning, tool use, observation, verification,

recovery, execution state tracking, evaluation, an HTTP API, and

containerized deployment, with every claim backed by a passing test

suite and real evaluation runs against a local LLM.



\## Philosophy



The core design goal: \*\*the stronger the underlying LLM, the smarter

the system behaves — but even with a weak or unreliable model, the

system's correctness should not degrade.\*\* This is achieved by

strictly separating two concerns:



\- \*\*Intelligence\*\* (the LLM) decides \*what\* to do — which tool to

&#x20; call, how to break a task into steps, how to phrase an answer.

\- \*\*Reliability\*\* (a deterministic, rule-based verification and

&#x20; recovery layer) decides \*whether the result is actually correct\* —

&#x20; independent of the LLM's own confidence.



This separation is why Verification and Recovery were prioritized

early in this project, ahead of Planning and the API/Deployment work

that followed.



\## Architecture



User Task

&#x20;  |

&#x20;  v

Agent.run\_with\_planning()  \[optional multi-step entry point]

&#x20;  |

&#x20;  +--> Planner.create\_plan()  --> ordered list of steps

&#x20;  |

&#x20;  +--> for each step:

&#x20;  |

&#x20;  v

Agent.run\_with\_recovery()

&#x20;  |

&#x20;  +--> Agent.run\_with\_tools()  \[execution loop, max\_iterations bound]

&#x20;  |        |

&#x20;  |        +--> LLM.generate\_with\_tools()

&#x20;  |        |        |

&#x20;  |        |        +--> empty response? --> retried once, then

&#x20;  |        |        |                        controlled RuntimeError

&#x20;  |        |        |

&#x20;  |        |        +--> tool call requested? --yes--> Tool Registry

&#x20;  |        |        |                                     |

&#x20;  |        |        |                              Tool execution

&#x20;  |        |        |                          (auto-retried once on

&#x20;  |        |        |                           transient failure)

&#x20;  |        |        |                                     |

&#x20;  |        |        |                              ToolResult

&#x20;  |        |        |                                     |

&#x20;  |        |        +<---- observation fed back into context

&#x20;  |        |

&#x20;  |        +--> final LLM response (no more tool calls)

&#x20;  |

&#x20;  +--> Verifier.verify(task, response)

&#x20;  |        |

&#x20;  |        +--> PASS --> return VerifiedAgentResult

&#x20;  |        |

&#x20;  |        +--> FAIL --> feedback (including full attempt history)

&#x20;  |                       injected into next retry's task

&#x20;  |                              |

&#x20;  |                              v

&#x20;  |                    (loop back to run\_with\_tools,

&#x20;  |                     up to max\_retries)

&#x20;  |

&#x20;  v

ExecutionState (iterations, tool\_calls\_made, tool\_retries,

&#x20;  empty\_response\_retries, retries, steps\_executed)

&#x20;  attached to every result for observability



FastAPI (app/api.py)

&#x20;  POST /run    --> runs a task through run\_with\_recovery

&#x20;  GET  /health --> liveness check



Docker

&#x20;  Containerizes the FastAPI app; connects to Ollama running on the

&#x20;  host machine via host.docker.internal (the LLM itself is not

&#x20;  containerized — it's treated as an external dependency).

`



\## Implemented Features



| Layer | Status | Notes |

|---|---|---|

| LLM Interface | Done | Provider-agnostic (`LLMClient`), Ollama implementation |

| Tool System | Done | `Tool` ABC, `ToolRegistry`, 2 tools registered |

| Calculator Tool | Done | AST-based safe evaluation, no `eval()` |

| Text Analyzer Tool | Done | Word/character counting |

| Execution Loop | Done | Bounded iterations, tool-call observation fed back to LLM |

| Planning | Done | `Planner` breaks a task into ordered steps via a constrained LLM prompt format; `run\_with\_planning` executes them in sequence with prior results as context |

\[9/27/2026 2:50 PM] Myself IR\_MCI: | Verification | Done | CalculatorVerifier — rule-based, AST-evaluated, extracts the correct number from natural-language LLM output (handles reasoning chains, LaTeX-style notation, parenthesized expressions) |

| Recovery (verification failures) | Done | Retry with injected failure reason; tracks full attempt history; explicitly flags repeated wrong answers to the LLM |

| Recovery (tool failures) | Done | Transient tool errors are retried once before being reported, independent of the verification/recovery layer |

| Recovery (empty LLM responses) | Done | Some models exhaust their output budget on internal "thinking" before producing content; empty final responses are retried once, then raise a controlled error rather than being treated as valid |

| Execution State | Done | Tracks iterations, tool calls, tool retries, empty-response retries, verification retries, and steps executed per run |

| Evaluation | Done | Evaluator + EvaluationCase harness; measures real Task Success Rate against a live LLM |

| API | Done | FastAPI with POST /run and GET /health, dependency-injected agent for testability |

| Deployment | Done | Dockerfile (python:3.11-slim), connects to host-run Ollama via host.docker.internal |



\## Evaluation Results



Running the evaluation harness (scripts/run\_evaluation.py) against a

live Ollama instance (qwen3.5:latest) on a 4-case arithmetic

benchmark:





Total cases:   4

Passed cases:  4

Success rate:  100.0%

This number is the result of real debugging, not a lucky first run.

The initial run scored 50% and surfaced two real bugs in

CalculatorVerifier — it was reading the first number in the LLM's

answer instead of the last, and didn't support parenthesized

expressions. Both were fixed with a test-first approach and verified

against the live model, not just mocks.



A second round of evaluation, this time against run\_with\_planning

(scripts/run\_planning\_evaluation.py), surfaced a deeper reliability

issue: this particular model can spend its entire output token budget

on internal "thinking" before ever producing final content, returning

an empty response. This was root-caused by inspecting the raw Ollama

API response (not guessed at), fixed by increasing the output token

budget and adding explicit empty-response retry logic in the

execution loop, then confirmed stable across repeated live runs.



This is the intended purpose of the evaluation harness: catching

reliability bugs that unit tests with mocks cannot.



\## Project Structure





app/

&#x20; agent/

&#x20;   agent.py         Agent: run, run\_tool, run\_with\_tools,

&#x20;                     run\_with\_verification, run\_with\_recovery,

&#x20;                     run\_with\_planning

&#x20;   models.py         ToolResult

&#x20;   state.py          ExecutionState

&#x20;   planning.py        Planner, Plan

&#x20;   verification.py   Verifier, VerificationResult, CalculatorVerifier

&#x20; llm/

&#x20;   client.py, models.py, ollama.py

&#x20; tools/

&#x20;   base.py           Tool interface

&#x20;   calculator.py     CalculatorTool

&#x20;   text\_analyzer.py  TextAnalyzerTool

&#x20;   registry.py       ToolRegistry

&#x20; evaluation/

&#x20;   evaluator.py      Evaluator, EvaluationCase, EvaluationResult

&#x20;   cases.py          Benchmark case definitions

&#x20; api.py              FastAPI app (/run, /health)

scripts/

&#x20; run\_evaluation.py           Runs the calculator evaluation harness

&#x20; run\_planning\_evaluation.py  Runs the planning evaluation harness

tests/                87+ tests covering every layer above

Dockerfile

.dockerignore



\## Setup



Requirements: Python 3.11+, \[Ollama](https://ollama.com) running

locally with a tool-calling-capable model pulled (this project was

developed against qwen3.5:latest).



\### Local



git clone <repo-url>

cd AI\_AGENTIC\_SYSTEM

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt

Copy `.env.example to .env and adjust if needed:



``

LLM\_PROVIDER=ollama

OLLAMA\_BASE\_URL=http://localhost:11434

OLLAMA\_MODEL=qwen3.5:latest



Run the API:



powershell

uvicorn app.api:app --reload

`

\[9/27/2026 2:50 PM] Myself IR\_MCI: Then open http://127.0.0.1:8000/docs for the interactive API

explorer.



\### Docker



Make sure Ollama is running on the host machine, then:



docker build -t agentic-ai-system .

docker run -p 8000:8000 agentic-ai-system

The container reaches the host's Ollama instance via

host.docker.internal (works on Docker Desktop for Windows/Mac).



\## Running Tests



pytest -v

\## Running the Evaluation Harnesses



Make sure Ollama is running and the configured model is pulled, then:



python -m scripts.run\_evaluation

python -m scripts.run\_planning\_evaluation

\## Roadmap



This project follows a staged roadmap. Verification, Recovery, State,

and Evaluation were deliberately prioritized ahead of Planning and

API/Deployment work, because reliability was judged foundational

rather than a later addition.



\- \[x] V0 — single-tool agent with tool calling and observation

\- \[x] V0.1 — multi-tool agent (2 tools registered)

\- \[x] V0.2 — planning and task decomposition

\- \[x] V1 — verification, recovery (verification/tool/empty-response),

&#x20;     execution state, evaluation, FastAPI, Docker deployment



All milestones above are tagged in the commit history

(v0.1.0-multi-tool, v0.2.0-planning, v1.0.0).



\## License



MIT

