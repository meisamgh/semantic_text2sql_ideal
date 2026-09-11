Act as a senior AI/ML architect, staff engineer, and critical code reviewer.

Analyze this project deeply,

Do NOT evaluate the project only from the README. Read and understand the actual source code, architecture, tests, configuration, benchmarks, prompts, database layer, retrieval components, model adapters, recovery logic, UI/API code, commit history, and any open/closed issues or pull requests available to you.

Your goal is to determine what this system really does, how strong it is, where it can fail, and what would make it significantly better.

Please perform the analysis in these areas:

1. ARCHITECTURE

* Reconstruct the real end-to-end workflow from the code.
* Identify deterministic components, ML components, LLM components, and agentic components.
* Identify where state is stored and passed.
* Identify unnecessary complexity or duplicated logic.
* Point out places where the README and implementation disagree.

2. TEXT-TO-SQL QUALITY
   Evaluate:

* schema linking
* table selection
* column selection
* value retrieval
* historical-query retrieval
* business glossary handling
* formula handling
* PK/FK and bridge-table restoration
* grain/cardinality/fanout handling
* date/time semantics
* SQL generation
* SQL validation
* execution feedback
* repair/recovery
* semantic correctness detection

Pay special attention to SQL that:

* executes successfully
* returns plausible results
* but answers the wrong business question.

Explain which failure modes the current architecture cannot reliably detect.

3. RETRIEVAL
   Analyze the BM25, dense retrieval, value matching, RRF, reranking, historical retrieval, and schema profiling pipeline.

Determine:

* which components materially help
* which components probably add complexity without enough gain
* failure modes
* retrieval leakage risks
* whether table-first / column-first retrieval is optimal
* how retrieval could be improved without significantly increasing token cost.

4. LLM USAGE AND PROMPTS
   Read the actual prompts.

For every LLM call explain:

* purpose
* input
* output
* approximate context size
* whether the call is necessary
* whether the same task could be deterministic
* possible hallucination/failure modes

Identify opportunities to reduce:

* token consumption
* number of calls
* latency
  without reducing accuracy.

5. CANDIDATE GENERATION

Compare the current mostly-single-candidate architecture with approaches such as:

* one candidate
* self-consistency
* multiple samples from one prompt
* different reasoning prompts
* Divide-and-Conquer generation
* Skeleton/Plan generation
* retrieved-example / ICL generation
* multiple different models
* execution-result clustering

Determine whether conditional multi-candidate generation would improve this project.

Do NOT simply recommend generating many candidates for every query.

Design a difficulty/uncertainty gate so:

* simple queries use one candidate
* ambiguous/complex queries may use multiple candidates
* extra LLM/database cost is incurred only when justified.

6. COST

Analyze the current cost structure.

Estimate what determines:

* input tokens
* output tokens
* number of LLM calls
* recovery probability
* latency
* database execution cost

Propose a pre-generation cost/risk estimator.

Show how the project could predict:

expected_llm_cost
expected_latency
expected_repair_probability
expected_database_cost

before expensive execution.

7. DATABASE AND SECURITY
   Review:

* SQL read-only guarantees
* SQLGlot validation
* SQLite security
* PostgreSQL security
* query timeout behavior
* expensive query protection
* SQL injection / unsafe SQL risks
* DB permissions
* concurrent query risks

Identify anything that is safe in a benchmark environment but unsafe in production.

8. BUG HUNTING

Do a real code review.

Find:

* actual bugs
* incorrect wiring
* configuration drift
* dead code
* incorrect assumptions
* timeout bugs
* race/concurrency issues
* memory leaks
* inconsistent interfaces
* benchmark bugs
* evaluation bugs
* provider/model routing errors
* mismatches between tests and production behavior

For every finding provide:

Severity: P0 / P1 / P2 / P3
File:
Relevant function/class:
Problem:
Why it matters:
Recommended fix:

Do not invent bugs. Distinguish confirmed bugs from suspected risks.





11. COMPARE WITH STRONG TEXT-TO-SQL SYSTEMS

Using current publicly available information if internet access is available, compare this architecture with strong systems such as:

DeepEye-SQL
XiYan-SQL
SIRIUS-SQL
Agentar-Scale-SQL
and other relevant current systems.

Do not compare only leaderboard scores.

Compare architecture in terms of:

semantic accuracy
schema linking
candidate diversity
verification
business semantics
cost
latency
database load
security
explainability
production readiness.

Identify specific ideas worth borrowing and ideas that should NOT be copied.


14. FINAL VERDICT

At the end provide:

A. Architecture score /10
B. SQL accuracy potential /10
C. Retrieval quality /10
D. Cost efficiency /10
E. Production readiness /10
F. Code quality /10
G. Security /10
H. Novelty /10

Then answer:

"What are the 5 biggest reasons this system will fail on difficult real-world questions?"

"What are the 5 strongest aspects of this architecture?"

"What are the 10 highest-impact improvements?"

Rank improvements by:

Impact
Engineering effort
Token/cost increase
Expected accuracy gain
Production value

Finally give me a recommended V6 architecture.

Keep the existing design philosophy where it makes sense:

DETERMINISTIC FIRST
↓
RETRIEVE ONLY NECESSARY CONTEXT
↓
MINIMIZE TOKEN USAGE
↓
ONE MODEL CALL WHEN POSSIBLE
↓
ESCALATE COMPUTE ONLY WHEN UNCERTAINTY JUSTIFIES IT
↓
BOUNDED RECOVERY
↓
AUDITABLE / PRODUCTION-SAFE BEHAVIOR

Do not recommend adding agents simply because agents are fashionable.

For every important conclusion, cite the exact source file/function or repository evidence that led you to that conclusion.

Clearly distinguish:
CONFIRMED FROM CODE
INFERENCE
RECOMMENDATION
