# 🔥 CodeFuse-Agent: Achieving 61.67% on SWE-bench Lite via Trajectory-Aware Test Time Scaling

## 1. Abstract
We present CodeFuse-Agent, an AI agent designed to tackle with software engineering challenges, achieving **61.67% resolution rate** on SWE-bench Lite, establishing a new state-of-the-art. Our approach comprises two stages: 

(1) Multi-trajectory patch generation by our core agent framework **CodeFuse-Agent**

(2) **Trajectory-Aware Test-Time Scaling (TTS)** which performs systematic candidate selection by cross-validating patches against self-generated test cases consolidated from historical debugging trajectories. 

By decoupling generation from verification and exploiting the collective debugging artifacts across trajectories, **CodeFuse-Agent** substantially improves patch selection accuracy.

We fully open-source **CodeFuse-Agent** to facilitate reproducibility and benefit the broader research community. By lowering the barrier to entry, we hope to accelerate collective progress in building more capable AI coding agents.

## 2. Introduction
Automated program repair remains a critical challenge in software engineering. While LLM-based agents continue to improve, complex issues often cannot be resolved in a single attempt. Increasing the number of rollouts improves success probability, but introduces a new bottleneck: **how to reliably select the correct patch from multiple candidates?**

Current approaches predominantly rely on **LLM-as-Judge** methods, such as using LLMs to vote on candidate patches or even resorting to random selection when votes are tied. We argue that such approaches are inherently unstable and lack robustness—LLM judgments can be inconsistent across runs and sensitive to prompt variations. Furthermore, these approaches suffer from limited interpretability—the rationale provided by LLM judges lacks grounding in executable or testable evidence, making it hard to objectively validate the selection decision.

To explore this, we conducted a preliminary study on SWE-Bench-Verified (500 instances):

**Finding 1: Self-validation in a single rollout is unreliable.**

| | **Count** | **Empty Patch** | **with debugging process** | **Without Debugging process** |
| --- | --- | --- | --- | --- |
| unresolved | 174 | 4 | 157 | 13 |
| resolved | 326 | 0 | 283 | 43 |
| All Instance | 500 | 4 | 440 | 56 |


**Analysis Base on SWE-Bench-Verified**

We observed that 88% of instances (440/500) included self-generated tests during the debugging process. However, 35.7% of these (157/440) ultimately failed ground-truth evaluation—despite the agent believing its patch had passed it's own test cases. This indicates that tests from a single trajectory can be incomplete or incorrect, leading to false confidence.

**Finding 2: Multiple rollouts reveal stronger collective potential.**

![](./images/oracle_vs_adversary_performance.png)

**Influence of attempt count to overall performance**

A single rollout achieves 65.2% success rate (326/500), while the best-of-N oracle reaches Y% (X/500)—a gap of Z percentage points. This demonstrates that correct patches do exist among the rollouts; the challenge lies in identifying them.



<font style="color:rgb(0, 0, 0);">These findings motivate our approach: </font>**<font style="color:rgb(0, 0, 0);">aggregate test cases from all rollouts to form a comprehensive test suite, and use it as executable, verifiable evidence for patch selection.</font>**<font style="color:rgb(0, 0, 0);"> </font>By pooling self-generated tests from all trajectories and selecting the patch with the highest pass rate, we leverage the model's latent ability to produce good tests—even when individual rollouts are inconsistent.

To summary, in this work, we present two contributions: (1) **CodeFuse-Agent(CFuse)**, a lightweight, research-oriented agent framework for code generation, and (2) **Trajectory-Aware Test-Time Scaling (TTS)**, a verification mechanism that aggregates self-generated tests from all trajectories for cross-trajectory validation. Together, they achieve **61.67% resolution rate** on SWE-bench Lite.

## 3. Methodology
### 3.1 Stage 1: Multi-Trajectory Patch Generation
The first stage employs our agent framework to generate diverse candidate patches through multiple independent trajectories.

#### 3.1.1 CodeFuse Agent Architecture
CodeFuse-Agent(CFuse) is a lightweight, cleanly-architected agent framework designed for research and experimentation. It is fully open-source and can be installed with a single `pip install` command, providing a complete yet minimal toolset for code related task. We open-source CFuse to facilitate reproducible research and encourage further exploration of LLM-based coding agents.

![](./images/CFuse_Architecture.png)

| **Layer** | **Responsibility** |
| :--- | :--- |
| Interaction | Terminal UI / Headless /HTTP modes |
| Agent Loop | Core lifecycle: LLM interaction, tool dispatch, iteration control |
| Context Engine | Message history, environment context, compression, prompt assembly |
| LLM Provider | Multi-LLM support (OpenAI, Anthropic, Gemini, etc.) |
| Tool Execution | 6 built-in tools + remote execution |
| Observability | Trajectory logs, execution metrics, cost tracking |


**Configurable Agent Profiles**

Agent behavior is defined through declarative Markdown profiles(system prompt, tools, model etc.), enabling quick switching of system prompts and tool subsets without code changes.

**Dual Execution Modes**

+ Local Mode: Execute tool calls directly in the local environment
+ HTTP Mode: Serve as a tool execution backend or delegate calls to remote sandboxes

This decoupling of agent decisions from environment execution makes CFuse suitable as scaffolding for RL training pipelines.

**Built-in Tools**

The framework provides six built-in tools for code exploration and modification:

+ read_file: Read file contents with optional line range selection
+ write_file: Create or overwrite files
+ edit_file: Perform edits via search-and-replace
+ grep: Fast code search powered by ripgrep
+ glob: File discovery using glob patterns
+ bash: Execute shell commands with timeout control



### 3.2 Stage 2: Trajectory-Aware Test Time Scaling
![](./images/TTS_framework.jpg)

Building upon the trajectories from Stage 1, this stage performs systematic verification and selection through three sequential components. 

#### 3.2.1 Test Case Consolidation
Rather than designing yet another heuristic-driven test generation agent, we propose reframing the problem: we introduce a **Test Consolidate Agent** that consolidate debugging experience from agents' own debugging trajectory into a single executable test file. 

<font style="color:rgb(13, 18, 57);">Moreover, to mitigate the issue of excessively long context windows, we first extracted only the tool invocation content from those steps in the agent's execution trajectory that are relevant to debugging. Furthermore, we adopted a sliding window approach, processing only a window of </font>_**<font style="color:rgb(13, 18, 57);">N</font>**_<font style="color:rgb(13, 18, 57);"> consecutive trajecory，the following formulate will explain it:</font>

Let there be $ M $ agent execution trajectories:

$$ \mathcal{T} = \\{ T^{(1)}, T^{(2)}, \dots, T^{(M)} \\}, $$

<font style="color:rgb(13, 18, 57);">where the </font>$ m $<font style="color:rgb(13, 18, 57);">-th trajectory is denoted as </font>$ T^{(m)} = \\{ s^{(m)}_1, s^{(m)}_2, \dots, s^{(m)}_{L_m} \\} $<font style="color:rgb(13, 18, 57);">, and </font>$ L_m $<font style="color:rgb(13, 18, 57);"> is its length. </font>$ s^{m}_i $<font style="color:rgb(13, 18, 57);">is the step it takes in </font>$ i_{th} $<font style="color:rgb(13, 18, 57);">iteration round. For each trajectory </font>$ T^{(m)} $<font style="color:rgb(13, 18, 57);">, we perform the following steps:</font>

1. **Filter debugging-relevant steps**<font style="color:rgb(13, 18, 57);">:</font>

$$ D^{(m)} = \\{ s^{(m)}_i \in T^{(m)} \mid \text{IsDebugRelevant}(s^{(m)}_i) = \text{true} \\}. $$

2. **Extract tool invocation content**<font style="color:rgb(13, 18, 57);"> from those steps:</font>

> <font style="color:rgb(13, 18, 57);">where </font>$ K_m = |C^{(m)}| $<font style="color:rgb(13, 18, 57);"> denotes the number of extracted tool invocations.</font>
>

$$ C^{(m)} = \\{ c^{(m)}_j = \text{ExtractToolInvocation}(s) \mid s \in D^{(m)} \\}, $$

3. **Apply a sliding window**<font style="color:rgb(13, 18, 57);"> of fixed size </font>$ N $<font style="color:rgb(13, 18, 57);"> over </font>$ C^{(m)} $<font style="color:rgb(13, 18, 57);"> to generate contextual segments:</font>

$$ \mathcal{W}^{(m)} = \\{ W^{(m)}_k = ( c^{(m)}_k, c^{(m)}_{k+1}, \dots, c^{(m)}_{k+N-1} ) \mid k = 0, 1, \dots, \max(0, K_m - N) \\}. $$

4. <font style="color:rgb(13, 18, 57);">(If </font>$ K_m < N $<font style="color:rgb(13, 18, 57);">, implementations may either skip the trajectory or treat the entire </font>$ C^{(m)} $<font style="color:rgb(13, 18, 57);"> as a single window.)</font>
5. **Enrich the corresponding test file**<font style="color:rgb(13, 18, 57);"> </font>$ f^{(m)} $<font style="color:rgb(13, 18, 57);"> using each window:</font>

$$ \mathcal{E}^{(m)} = \\{ \text{ENRICH}(f^{(m)}, W^{(m)}_k) \mid W^{(m)}_k \in \mathcal{W}^{(m)} \\}. $$

<font style="color:rgb(13, 18, 57);">The final output is the union of all enriched test cases across trajectories:</font>

$$ \mathcal{E} = \bigcup_{m=1}^{M} \mathcal{E}^{(m)}. $$

<font style="color:rgb(13, 18, 57);">This approach ensures that context length remains bounded (by </font>$ N $<font style="color:rgb(13, 18, 57);">) while preserving only the most relevant tool interactions for debugging within each individual execution trace.</font>

#### 3.2.2 Cross-Validation and Filtering
Each candidate patch $ p_i $ is executed against the unified test suite $ T = \\{t_1, t_2, ..., t_n\\} $. We compute:

$$ \text{score}(p_i) = \sum_{j=1}^{n} \mathbb{1}[\text{pass}(p_i, t_j)] $$

Patches are ranked by their pass counts. The **top-K candidates** with highest scores proceed to the final selection stage. <font style="color:rgb(13, 18, 57);">Meanwhile, we employ a </font>**<font style="color:rgb(13, 18, 57);">Test Evaluation Agent</font>**<font style="color:rgb(13, 18, 57);"> to execute unit tests and report the final pass/fail status, thereby mitigating potential test execution failures or compilation errors caused by long-tail engineering bugs.</font>

## 4. Main Results
### 4.1 Experiment Setup

For each issue, we generate **4 candidate trajectories** using two model configurations:

+ Claude Sonnet 4: 2 trajectories
+ Claude Sonnet 4.5: 2 trajectories

All temperature are set to 0, Each trajectory executes within the official SWE-bench Docker environment. The agent iteratively explores the codebase, formulates hypotheses, and produces a candidate patch.

<font style="color:rgb(13, 18, 57);">In our implementation of TTS, all agents(Test Consolidate Agent & Test Evaluation Agent) are constructed based on </font>**<font style="color:rgb(13, 18, 57);">CFuse</font>**<font style="color:rgb(13, 18, 57);">, with variations across tasks in terms of the system prompt and the set of tools available for use.</font>

**Single Attempt Results:**

| Base Model | Resolved |
|------------|----------|
| Claude-Sonnet-4 | 54.67% |
| Claude-Sonnet-4 | 54% |
| Claude-Sonnet-4.5 | 60% |
| Claude-Sonnet-4.5 | 61% |

**Multi-Trajectory Statistics (Combined):**

| Oracle | Adversary | Average@1 | Average@2 | Average@3 | TTS Rank@1 | TTS Rank@2 | TTS Rank@3 |
|--------|-----------|-----------|-----------|-----------|------------|------------|------------|
| 68.67% | 47%       | 57.67%    | 64%       | 65.33%    | **61.67%** | 65%         | 66.33%     |


> Oracle: an instance is considered passed if any given patches passes all test cases.
>
> Adversary: an instance is considered passed if all given patches passes all test cases.
>
> Average@K: an instance is considered passed if any randomly sampled K patches passes all test cases.
>
> Test Case Consolidation: Apply Test Case Consolidation to all patches, and rank the patch according to pass rate
>

### 4.2 Observations & Analysis
**Single vs. Multiple attempts**

1. **Strong Single-Attempt Performance with Inherent Variability**<font style="color:rgb(13, 18, 57);">: Claude 4.5 achieves high resolution rates (60–61%) in a single attempt, but fluctuations indicate stochasticity in its reasoning, suggesting some problems inherently require multiple tries.</font>
2. **Significant Portion of Problems Are Not Solvable in One Attempt**<font style="color:rgb(13, 18, 57);">: </font><font style="color:rgb(13, 18, 57);">The gap between </font>**<font style="color:rgb(13, 18, 57);">Oracle</font>**<font style="color:rgb(13, 18, 57);"> (68.67% solved with any attempts) and </font>**<font style="color:rgb(13, 18, 57);">Adversary</font>**<font style="color:rgb(13, 18, 57);"> (47% solved with all attempt) results indicates that </font>**<font style="color:rgb(13, 18, 57);">21.67% of problems are solvable in principle but not reliably resolved in a single attempt</font>**<font style="color:rgb(13, 18, 57);">, highlighting the role of randomness in successful inference.</font>
3. **Multiple Attempts Substantially Boost Success Rates**<font style="color:rgb(13, 18, 57);">: Allowing up to four attempts increases overall solvability to 68.67%, with metrics like Average@k confirming a consistent positive correlation between allowed attempts and task resolution.</font>

> <font style="color:rgb(13, 18, 57);">insight: We require a robust and systematic approach to reliably derive a correct solution from multiple inference attempts—this necessity constitutes a primary motivation for implementing test case consolidation.</font>
>

---

**Test Case Consolidation Gains**

1. <font style="color:rgb(13, 18, 57);">The top-ranked patch selected via pass-rate–based consolidation (Rank@1) significantly outperforms both the average single-attempt success rate (Average@1) and the best-known single-attempt result of Claude 4.5, demonstrating its effectiveness in identifying high-quality solutions.</font>
2. <font style="color:rgb(13, 18, 57);">The performance gain is not limited to the top candidate—</font>**<font style="color:rgb(13, 18, 57);">Rank@2</font>**<font style="color:rgb(13, 18, 57);"> (65%) and </font>**<font style="color:rgb(13, 18, 57);">Rank@3 </font>**<font style="color:rgb(13, 18, 57);">(66.33%) also markedly exceed </font>**<font style="color:rgb(13, 18, 57);">Average@2</font>**<font style="color:rgb(13, 18, 57);"> (64%) and </font>**<font style="color:rgb(13, 18, 57);">Average@3</font>**<font style="color:rgb(13, 18, 57);"> (65.33%), indicating that test case consolidation yields more reliable and higher-quality candidate rankings across multiple positions.</font>
3. <font style="color:rgb(17, 17, 51);">Reranking based on Test Case Consolidation narrows the gap with the oracle; however, relying solely on </font>**<font style="color:rgb(17, 17, 51);">Rank@1</font>**<font style="color:rgb(17, 17, 51);"> still leaves a noticeable performance gap. We leave for future work the exploration of how to further identify the best patch from </font>**<font style="color:rgb(17, 17, 51);">Rank@2</font>**<font style="color:rgb(17, 17, 51);"> or even </font>**<font style="color:rgb(17, 17, 51);">Rank@3</font>**<font style="color:rgb(17, 17, 51);"> candidates.</font>

## 5. Conclusion
We presented CodeFuse-Agent, a system achieving 61.67% resolution rate on SWE-bench Lite through Trajectory-Aware Test Time Scaling. Our key contribution is demonstrating that agent debugging artifacts—particularly self-generated tests—provide valuable signals for patch selection that complement traditional execution-based validation. The decoupling of diverse generation from systematic verification offers a principled framework for scaling test-time compute in code repair tasks.

---