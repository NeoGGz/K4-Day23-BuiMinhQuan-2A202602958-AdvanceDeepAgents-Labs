# A Comprehensive Survey on Large Language Model Agents and Tool Use

## TL;DR
- Large language model (LLM) agents combine parametric reasoning with external capabilities through foundational execution paradigms such as self-supervised API calling, interleaved reasoning-acting (ReAct), and chain-of-abstraction reasoning [1][2][3].
- Massive function-calling datasets and evaluation benchmarks like APIBench, ToolBench, WebArena, and AgentDojo reveal critical capabilities alongside severe security risks including indirect prompt injection [4][5][6][7].
- Modern multi-agent orchestration frameworks (such as LangGraph, CrewAI, and Microsoft Agent Framework) leverage stateful graph structures, dynamic collaboration priors, and multi-agent debate to automate complex software and workflow tasks [8][9][10][11].
- Recent breakthroughs emphasize reinforcement learning for self-evolving tool learning, schema-grounded multi-task honing (SMITH), and hypergraph-guided planning over large-scale tool ecosystems [12][13][14].

## Background
The paradigm of Large Language Model (LLM) agents represents a foundational shift from static text generation to autonomous, goal-directed problem solving. While standard LLMs are constrained by frozen parametric memory and static training cutoffs, equipping models with external tool use—ranging from calculators and search engines to complex RESTful APIs and software interpreters—bridges the gap between abstract reasoning and grounded execution. Foundational milestones such as Toolformer demonstrated that language models could learn to annotate and invoke APIs autonomously via self-supervised perplexity filtering [1]. Simultaneously, the ReAct paradigm established that interleaving verbal reasoning traces ("thoughts") with executable actions ("acts") significantly enhances interpretability and mitigates hallucination in interactive environments [2]. The significance of agentic tool use lies in its capacity to transform passive language models into active computational agents capable of executing multi-step workflows, retrieving external knowledge in real time, and operating software tools with human-like autonomy.

## Foundational Execution Paradigms and Reasoning Architectures
The architectural evolution of LLM tool use has progressed from basic zero-shot function calling to sophisticated multi-step reasoning frameworks. Toolformer pioneered self-supervised tool integration by prompting models to insert API calls into raw text corpora and retaining those that minimized future token perplexity [1]. To overcome the limitations of isolated reasoning or ungrounded action, the ReAct framework synergized verbal thought generation with tool execution, enabling models to dynamically adjust their plans based on intermediate tool feedback [2]. 

Recent advancements address efficiency bottlenecks inherent in sequential tool interactions. Chain-of-Abstraction (CoA) reasoning trains models to decode reasoning chains containing abstract placeholders before executing parallelized tool calls, thereby reducing latency and improving generalization across domain knowledge shifts [3]. Furthermore, contrastive optimization frameworks such as AVATAR utilize actor-comparator loops to refine prompt trajectories through comparative analysis of positive and negative execution traces, steering agents away from local optimization traps during complex multi-stage tasks [15]. To scale beyond fixed instruction templates, recent work introduces schema-grounded reinforcement learning (e.g., SMITH) and agent-native tool primitives that wrap APIs in natural language interfaces, bridging the gap between rigid schema definitions and flexible execution [16][12].

## Benchmarks, Evaluation Frameworks, and Execution Environments
Evaluating agentic tool use requires rigorous benchmarks that test functional correctness, multi-turn dialogue adaptation, and realistic environment interaction. Gorilla and APIBench established early foundations for assessing API selection across hundreds of domains, highlighting the inadequacy of static abstract syntax tree (AST) matching and emphasizing execution-based evaluation [4]. Expanding upon single-turn function calling, ToolBench introduced ToolLLM alongside a depth-first search based decision tree (DFSDT) planner to evaluate multi-step tool navigation across over 16,000 real-world APIs, paired with automated machine evaluation (ToolEval) for pass and win rates [5]. API-Bank and WebArena further broadened evaluation scopes by providing multi-turn dialog benchmarks and self-hostable, end-to-end enterprise web environments that simulate realistic e-commerce, content management, and administrative workflows [17][6].

Comprehensive surveys on agent evaluation categorize metrics into task success rates, tool selection accuracy, and adversarial robustness [18]. However, long-horizon planning in massive tool ecosystems remains a critical bottleneck, as demonstrated by PlanBench-XL, which evaluates agent resilience under dynamic disruptions and incomplete observability [19].

## Safety, Security Vulnerabilities, and Defensive Alignment
As LLM agents gain autonomous execution privileges and access to private APIs, databases, and enterprise systems, security vulnerabilities have emerged as a paramount concern. A critical threat vector is indirect prompt injection (IPI), wherein malicious instructions hidden within untrusted tool outputs (such as retrieved web pages, emails, or API payloads) hijack the agent's control flow and force unauthorized actions [7]. 

To quantify and mitigate these risks, dynamic evaluation suites like AgentDojo benchmark prompt injection susceptibility alongside defense mechanisms such as taint tracking and tool output sanitization [7]. Recent audits of agentic security harnesses emphasize the necessity of rigorous validation standards for tool-call mediation under adversarial conditions, addressing critical defects such as silent payload non-delivery and conflated false-rejection rates [20]. Complementary safety benchmarks including AgentHarm, Agent Security Bench (ASB), and SafeAgentBench systematically evaluate harmful behavior detection and adversarial robustness across diverse agent architectures [18].

## Multi-Agent Collaboration and Orchestration Frameworks
Scaling agentic workflows beyond single-agent capabilities has driven the development of sophisticated multi-agent orchestration frameworks. Production-grade platforms such as CrewAI, LangGraph, Microsoft Agent Framework, and BeeAI provide stateful, graph-based execution architectures (inspired by dataflow models) that support central persistence, fine-grained control flow, and human-in-the-loop validation [9][10]. 

Advanced orchestration research focuses on overcoming memoryless execution silos through adaptive multi-agent coordination. WorkflowOps introduces frameworks that learn collaboration priors from historical workflow executions, dynamically expanding agent pools using transition probability matrices [8]. Similarly, dynamic hierarchical construction frameworks model multi-agent reasoning as partially observable Markov decision processes (POMDPs) that adapt graph composition in real time [21]. Structured frameworks like AgentSPEX provide dedicated domain-specific languages for modular state management [22], while self-organizing agent networks (SOAN) encapsulate structural units to enhance fault tolerance [23].

## Automated Workflow Optimization and Self-Evolving Agents
To eliminate manual prompt engineering and static workflow design, recent research has pivoted toward automated agent creation and self-evolving tool learning. DebFlow automates agentic workflow generation by employing multi-agent debate and proponent/opponent role-playing to explore directed graph execution spaces without getting trapped in local optima [11]. In software engineering domains, SWE-Debate combines static code dependency traversal with competitive multi-agent debate to trace fault propagation across alternative call hierarchies and generate robust fix plans [24].

At the algorithmic level, zero-data self-evolving frameworks such as Tool-R0 train general-purpose tool-calling agents through self-play reinforcement learning without relying on pre-collected demonstration datasets [13]. Furthermore, hypergraph-guided planning frameworks (HyperAgent) model complex tool interdependencies at the schema level, significantly improving multi-tool composition and execution success in large-scale environments [14].

## Trends and Open Problems
- **Zero-Data Self-Evolution:** Transitioning from static supervised instruction tuning to self-play reinforcement learning (e.g., Tool-R0) to bootstrap agent capabilities without manual data collection remains a vibrant frontier [13].
- **Robust Tool-Call Mediation:** Defending against sophisticated indirect prompt injections and silent execution failures requires standardized runtime mediation and taint-tracking harnesses [7][20].
- **Long-Horizon Planning and Schema Scalability:** Managing thousands of APIs without catastrophic performance degradation necessitates advanced schema-grounded hypergraphs and abstract reasoning primitives [19][14].
- **Unified Multi-Agent Orchestration:** Moving from fragmented scripts to enterprise-grade stateful graphs and domain-specific execution languages (e.g., LangGraph, AgentSPEX) to ensure reliable state persistence and human oversight [22][9].
- **Joint Tool Creation and Execution:** Co-optimizing tool synthesis and tool invocation within unified reinforcement learning policies (e.g., SMITH) to bridge the gap between static API catalogs and dynamic agent needs [12].

## References
[1] Toolformer: Language Models Can Teach Themselves to Use Tools. web. https://proceedings.neurips.cc/paper_files/paper/2023/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf (2023-02-09)
[2] ReAct: Synergizing Reasoning and Acting in Language Models. web. https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/ (2022-11-08)
[3] Efficient Tool Use with Chain-of-Abstraction Reasoning. web. https://aclanthology.org/2025.coling-main.185.pdf (2025-01-01)
[4] Gorilla: Large Language Model Connected with Massive APIs. arxiv. https://arxiv.org/abs/2305.15334 (2023-05-15)
[5] ToolLLM: Facilitating Large Language Model to Master 16,000+ Real-world APIs. arxiv. https://arxiv.org/abs/2307.16789 (2023-07-27)
[6] WebArena: A Realistic Web Environment for Building Autonomous Agents. arxiv. https://arxiv.org/abs/2307.05525 (2023-07-11)
[7] AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. arxiv. https://arxiv.org/abs/2406.13352 (2024-06-18)
[8] WorkflowOps: Learning Agent Collaboration Priors for Multi-Agent Workflow Orchestration. arxiv. https://arxiv.org/abs/2610.07860 (2026-10-06)
[9] Comparing AI agent frameworks: CrewAI, LangGraph, and BeeAI. web. https://developer.ibm.com/articles/awb-comparing-ai-agent-frameworks-crewai-langgraph-and-beeai/ (2025-05-20)
[10] The best AI agent frameworks in 2026. web. https://www.langchain.com/resources/ai-agent-frameworks (2026-06-06)
[11] DebFlow: Automating Agent Creation via Agent Debate. web. https://arxiv.org/html/2503.23781 (2025-03-23)
[12] Joint Optimization of Tool Creation and Use for Large Language Model Agents. arxiv. https://arxiv.org/abs/2608.24571 (2026-08-25)
[13] Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data. hf-search. https://huggingface.co/papers/2602.21320 (2026-02-24)
[14] HyperAgent: Planning and Acting over Tool-Schema Hypergraphs for Tool-Use LLM Agents. hf-search. https://huggingface.co/papers/2608.02650 (2026-07-31)
[15] AVATAR: Optimizing LLM Agents for Tool Usage via Contrastive Reasoning. web. https://proceedings.neurips.cc/paper_files/paper/2024/file/2db8ce969b000fe0b3fb172490c33ce8-Paper-Conference.pdf (2024-12-01)
[16] Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives. arxiv. https://arxiv.org/abs/2609.01736 (2026-09-01)
[17] API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. arxiv. https://arxiv.org/abs/2304.08244 (2023-04-17)
[18] Evaluation and Benchmarking of LLM Agents: A Survey. arxiv. https://arxiv.org/abs/2507.21504 (2025-07-21)
[19] PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents in Large-Scale Tool Ecosystems. hf-search. https://huggingface.co/papers/2606.22388 (2026-06-21)
[20] Silent Failures in Agentic Security Evaluation: A Validated Harness for Tool-Call Mediation Under Indirect Prompt Injection. arxiv. https://arxiv.org/abs/2609.32691 (2026-09-15)
[21] Dynamic Construction of Hierarchical Collaboration Graphs for LLM-Based Multi-Agent Reasoning. arxiv. https://arxiv.org/abs/2610.07835 (2026-10-06)
[22] AgentSPEX: An Agent SPecification and EXecution Language. hf-search. https://huggingface.co/papers/2604.13346 (2026-04-14)
[23] Self-Organizing Agent Network for LLM-based Workflow Automation. hf-search. https://huggingface.co/papers/2508.13732 (2025-08-19)
[24] SWE-Debate: Competitive Multi-Agent Debate for Software Issue Resolution. web. https://dl.acm.org/doi/pdf/10.1145/3744916.3787810 (2025-01-01)
