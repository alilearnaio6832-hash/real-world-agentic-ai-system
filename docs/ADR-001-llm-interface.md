 # ADR-001: Provider-Agnostic LLM Interface



 ## Status



Accepted



 ## Context



The Agentic AI System needs to communicate with a Large Language Model.



The system should not be tightly coupled to a specific LLM provider such as Ollama or an OpenAI-compatible API.



A direct dependency between the Agent and a provider would make future provider changes more difficult and would increase coupling between the Agent Core and infrastructure.



 ## Decision



We will introduce a provider-agnostic `LLMClient` interface.



The Agent will depend on the `LLMClient` abstraction rather than directly depending on a specific LLM provider.



The interface currently exposes:





generate(prompt: str) -> str



Concrete provider implementations will be added later.



 ## Consequences



 ### Positive

* Agent Core remains independent of the LLM provider.
* Providers can be replaced with limited changes.
* Testing can use mock LLM implementations.
* Architecture remains modular.
* Future provider integrations are easier.



 ### Negative



* Adds an abstraction layer.
* Provider-specific capabilities may require additional interface design later.



 ## Alternatives Considered



 ### Direct Ollama Integration 



Rejected for the initial architecture because it would tightly couple the Agent Core to Ollama.



 ### Agent Framework



Rejected for the initial version because the project should first establish its own core abstractions and execution model.

