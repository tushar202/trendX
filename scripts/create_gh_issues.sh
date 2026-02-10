#!/bin/bash

# Ensure gh is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) could not be found. Please install it first."
    exit 1
fi

echo "Creating issues from roadmap.md..."

# Issue 1: Smart Merge Deduplication
gh issue create --title "Enhancement: Smart Merge Deduplication" --body "
### Goal Description
The current deduplication logic discards duplicate items entirely, losing not just popularity signals but also unique content (claims, evidence) extracted from those duplicates. This change implements a 'Smart Merge' strategy to aggregate counts, alternative sources, and extracted insights.

### Proposed Changes
1. **Storage**: Add \`duplicate_count\` and \`related_urls\` to Item model.
2. **Logic**: Update \`dedupe_items\` to sum counts, merge claims/anchors, and track related URLs.
3. **Reporting**: Expose 'Verified across X sources' in the final report.
" --label "enhancement"

# Issue 2: LLM-Based Fact Extraction
gh issue create --title "Enhancement: LLM-Based Fact Extraction" --body "
### Problem
The current fact extraction in \`src/trendx/agents/research.py\` relies on a hardcoded list of keywords, missing valuable facts that don't match specific phrases.

### Solution
Replace or augment the keyword-based extraction with a lightweight LLM call or a more advanced NLP heuristic.

### Proposed Changes
1. **Logic**: Implement \`_extract_facts_llm\` using a small model (e.g., GPT-4o-mini).
2. **Configuration**: Add config option to enable/disable LLM extraction.
" --label "enhancement"

# Issue 3: Enhanced Trust Scoring
gh issue create --title "Enhancement: Enhanced Trust Scoring" --body "
### Problem
The current \`apply_trust_scores\` logic is too heuristic-based (ignores GitHub stars, relies on simple keywords).

### Solution
Incorporate quantitative metrics and qualitative LLM evaluation.

### Proposed Changes
1. **GitHub Signal**: Add \`log10(stars)\` factor.
2. **LLM Quality Score**: Normalize a 'technical_depth_score' (1-5) into the trust calculation.
3. **Author Authority**: Boost score for known high-quality organizations.
" --label "enhancement"

echo "Done! Issues created."
