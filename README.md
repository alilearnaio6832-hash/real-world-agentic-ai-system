\# Real-World Agentic AI System



A measurable, reliable, modular agentic AI system built from first

principles — not a LangChain wrapper. This project demonstrates real

agentic engineering: planning, tool use, observation, verification,

recovery, execution state tracking, and evaluation, with every claim

backed by a passing test suite and a real evaluation run against a

local LLM.



\## Philosophy



The core design goal: \*\*the stronger the underlying LLM, the smarter

the system behaves — but even with a weak or unreliable model, the

system's correctness should not degrade.\*\* This is achieved by

strictly separating two concerns:



\- Intelligence (the LLM) decides \*what\* to do — which tool to

&#x20; call, how to phrase an answer.

\- Reliability (a deterministic, rule-based verification and

&#x20; recovery layer) decides \*whether the result is actually correct\* —

&#x20; independent of the LLM's own confidence.



This separation is why Verification and Recovery were prioritized

early in this project, ahead of Planning and API/Deployment work.



\## Architecture





User Task

&#x20;  |

&#x20;  v

Agent.run\_with\_recovery()

&#x20;  |

&#x20;  +--> Agent.run\_with\_tools()  \[execution loop, max\_iterations bound]

&#x20;  |        |

&#x20;  |        +--> LLM.generate\_with\_tools()

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

ExecutionState (iterations, tool\_calls\_made, tool\_retries, retries)

&#x20;  attached to every VerifiedAgentResult for observability



\## Implemented Features



| Layer | Status | Notes |

|---|---|---|

| LLM Interface | Done | Provider-agnostic (LLMClient), Ollama implementation |

| Tool System | Done | Tool ABC, ToolRegistry, 2 tools registered |

| Calculator Tool | Done | AST-based safe evaluation, no eval() |

| Text Analyzer Tool | Done | Word/character counting |

| Execution Loop | Done | Bounded iterations, tool-call observation fed back to LLM |

| Verification | Done | CalculatorVerifier — rule-based, AST-evaluated, extracts the correct number from natural-language LLM output (handles reasoning chains, LaTeX-style notation, parenthesized expressions) |

| Recovery (verification failures) | Done | Retry with injected failure reason; tracks full attempt history; explicitly flags repeated wrong answers to the LLM |

\[9/22/2026 2:41 PM] Math: | Recovery (tool failures) | Done | Transient tool errors are retried once before being reported, independent of the verification/recovery layer |

| Execution State | Done | Tracks iterations, tool calls, tool retries, and verification retries per run |

| Evaluation | Done | Evaluator + EvaluationCase harness; measures real Task Success Rate against a live LLM |

| Planning / Task Decomposition | Not started | Tasks are currently executed as-is; no multi-step decomposition yet |

| API (FastAPI) | Not started | |

| Deployment (Docker, logging, monitoring) | Not started | |



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

against the live model, not just mocks. This is the intended purpose

of the evaluation harness: catching reliability bugs that unit tests

with mocks cannot.



\## Project Structure



app/

&#x20; agent/

&#x20;   agent.py         Agent: run, run\_tool, run\_with\_tools,

&#x20;                     run\_with\_verification, run\_with\_recovery

&#x20;   models.py         ToolResult

&#x20;   state.py          ExecutionState

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

scripts/

&#x20; run\_evaluation.py   Runs the evaluation harness against live Ollama

tests/                60+ tests covering every layer above



\## Setup



Requirements: Python 3.11+, \[Ollama](https://ollama.com) running

locally with a tool-calling-capable model pulled (this project was

developed against qwen3.5:latest).



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



Running Tests

pytest -v



Running the Evaluation Harness



Make sure Ollama is running and the configured model is pulled, then:



python -m scripts.run\_evaluation



\## Roadmap



This project follows a staged roadmap. Verification, Recovery, State,

and Evaluation were deliberately prioritized ahead of Planning and

API/Deployment work, because reliability was judged foundational

rather than a later addition.



\- \[x] V0 — single-tool agent with tool calling and observation

\- \[x] V0.1 — multi-tool agent (2 tools registered)

\- \[ ] V0.2 — planning and task decomposition (execution loop is done;

&#x20;     true multi-step planning is not yet implemented)

\- \[ ] V1 — adds a FastAPI interface and Docker-based deployment



\## License



MIT



