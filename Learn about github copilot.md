## <mark style="background: #FFB8EBA6;">Useful resources</mark> 

- https://github.com/features/copilot
- prompt engineering: https://github.com/dair-ai/Prompt-Engineering-Guide
- https://github.com/google-research/tuning_playbook
- https://openai.com/api/pricing/#faq-fine-tuning-pricing-calculation

- openAI [playground](https://beta.openai.com/account/billing/overview) with apis 

## <mark style="background: #FFB8EBA6;">How to work with copilot efficiently</mark> 

 <mark style="background: #BBFABBA6;">ideas forming + prompt crafting</mark> should be the most important thing, <mark style="background: #BBFABBA6;">extracting examples and tricks</mark> from the guides below
- microsoft prompt engineering [guide](https://microsoft.github.io/prompt-engineering/) 🔥🔥🔥🔥🔥 
- copilot guide 1: https://nira.com/github-copilot/
- copilot guide 2: https://shift.infinite.red/getting-the-most-from-github-copilot-8f7b32014748
- <mark style="background: #BBFABBA6;">what is and how to do prompt engineering</mark> [intro](https://amatriain.net/blog/PromptEngineering) 
- <mark style="background: #BBFABBA6;">openAI's guide</mark> on how to do prompting, [guide](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api) 
- openAI's codex <mark style="background: #BBFABBA6;">best practice for code-completion</mark> , [guide](https://beta.openai.com/docs/guides/code/best-practices) 
- openAI code [example](https://beta.openai.com/examples) 

## <mark style="background: #FFB8EBA6;">Papers </mark> 

#### <mark style="background: #FFB86CA6;">on behavior of using copilot</mark>  [tweet](https://twitter.com/DynamicWebPaige/status/1616835849848762369)

> revealed that, when solving a coding task with Copilot, programmers may spend a large fraction of total session time (34.3%) on just double-checking and editing Copilot suggestions, and spend more than half of the task time on Copilot related activities, together indicating that introducing Copilot into an IDE can significantly change user behavior. We also found that programmers may defer thoughts on suggestions for later and simply accept them when displayed. Programmers also spend a significant amount of time waiting for suggestions, possibly due to latency in Copilot’s current implementation. The CUPS diagram and patterns showed that, even if programmers defer thought, they end up spending time double-checking the suggestion later. We observed longer sequences such as cycles of accepts and verification, and sequences of consecutive accept with verification deferred at the end of these accepts. We also observed that accepts are correlated more with some CUPS than others, e.g., programmers are more likely to verify suggestions that they eventually accept. We observed that the cost of interaction is higher than previously expected. For example, programmers spend a great deal of time verifying after accepting; time to accept severely underestimates the time spent on verifying suggestions. We see as future work a leveraging of these initial insights and methodologies to inform further analysis of programmer-Copilot interaction and to develop policies and designs that enhance the interaction to provide greater value for programmers.

- "programmers spend a great deal of time verifying after accepting; time to accept severely underestimates the time spent on verifying suggestions."
- "a large fraction of total session time (34.3%) on just double-checking and editing Copilot suggestions, and spend more than half of the task time on Copilot related activities,"
- Is it because programmers at the moment have not seen the significance of using prompt with care?