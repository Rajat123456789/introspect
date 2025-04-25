Code files:

1. extract_analysis_from_html_report.py

Extracts analysis from HTML reports and saves them as `extracted_analysis/complete_analysis_*.md`. For e.g. [/extracted_analysis/complete_analysis_youtube.md](prompt-eng/extracted_analysis/complete_analysis_youtube.md)

Key arguments:
- Path to HTML report
- Path to output `complete_analysis.md`

2. prepare_history_context.py

Extracts key insights useful for introspection. Uses OpenAI o4-mini model for extraction from above generated `complete_analysis.md`. Saves the key insights at ``extracted_analysis/insights_summary_*.md`. For e.g. [/extracted_analysis/insights_summary_youtube.md](prompt-eng/extracted_analysis/insights_summary_youtube.md)

Key arguments:
- Path to complete_analysis.md or any other markdown file which contains the analysis
- OpenAI API key
- Path to output `insights_summary.md`

3. generate_prompt.py

Generates final prompt to send to the Introspective AI assistant. The prompt is built basis:

- LLM: openai_gpt4o, llama4_maverick, introspect_llm
- Context type: **insights** (generated above in `insights_summary.md`) , **raw** (cleaned, processed response from APIs)
- Context data: Youtube, Spotify, Health
- Introspect type: **mental** or **emotional**

The above are also the arguments for this script. The final prompt is saved at [/generated_prompt/latest_prompt.txt](prompt-eng/generated_prompt/latest_prompt.txt).

By default, without any arguments, this script generates prompt using: 
- Introspect LLM
- Context from insights
- Context data of all modalities
- Introspect type mental
