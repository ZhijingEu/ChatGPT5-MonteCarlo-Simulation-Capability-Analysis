# ChatGPT-5 Monte Carlo Simulation Capability Analysis
This repository contains the complete testing framework and results from evaluating ChatGPT-5's ability to run Monte Carlo simulations for risk analysis. The research was conducted in August 2025 following the release of ChatGPT-5.

## 📊 Overview

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/OrMVJEvWuV0/0.jpg)](https://www.youtube.com/watch?v=OrMVJEvWuV0)

https://www.youtube.com/watch?v=OrMVJEvWuV0

This repository contains the complete testing framework and results from evaluating ChatGPT-5's ability to run Monte Carlo simulations for risk analysis. The research was conducted in August 2025 following the release of ChatGPT-5.

**Key Finding**: ChatGPT-5 excels at cost risk analysis but struggles with complex schedule risk analysis implementation details.

## 📖 Full Article

Read the complete analysis: **[Can ChatGPT Run Monte Carlo Simulations?](https://medium.com/@zhijingeu/can-chatgpt-run-monte-carlo-simulations-5dcbcfa26514)**

## 🎯 Research Objectives

- Test ChatGPT-5's capability to perform Monte Carlo simulations
- Compare results with established tools (XLRisk, Python MCerp, Primavera Risk Analysis)
- Evaluate methodology selection and implementation quality
- Identify strengths and limitations for practical applications

## 🧪 Test Cases

### 1. Cost Risk Analysis - "Fantasy Island Holiday Budget"
- **Scenario**: Multi-variable vacation budget estimation
- **Complexity**: Moderate (uncertain ranges + discrete risk events + correlations)
- **ChatGPT-5 Performance**: ✅ **Excellent**
- **Key Success Factors**:
  - Intelligent methodology selection (Latin Hypercube sampling, Iman-Conover correlations)
  - Accurate probability distribution implementation
  - Results closely matched XLRisk and MCerp benchmarks

🔗 Chat Session Links https://chatgpt.com/share/68aae5d1-2798-800a-98b6-0f319832f3bd

### 2. Schedule Risk Analysis - "Garden Landscaping Project"  
- **Scenario**: Project schedule with uncertain activity durations
- **Complexity**: High (calendar logic + critical path + sensitivity analysis)
- **ChatGPT-5 Performance**: ❌ **Poor**
- **Key Issues Identified**:
  - Off-by-one date calculation errors
  - Incorrect handling of non-working days
  - Schedule Sensitivity Indices calculated as zero
  - Unreliable critical path identification

🔗 Chat Session Links https://chatgpt.com/share/68abf9e8-242c-800a-aa9d-84ee424f9622

## 📈 Results Summary

| Analysis Type | ChatGPT-5 Grade | Key Strengths | Major Limitations |
|---------------|----------------|---------------|-------------------|
| **Cost Risk** | A- | Methodology selection, correlation handling, statistical accuracy | Minor: Limited advanced distribution options |
| **Schedule Risk** | D+ | Conceptual understanding, code structure | Critical: Date arithmetic, calendar logic, sensitivity calculations |

## 🚨 Key Findings & Implications

### ✅ What Works Well
- **Simple to moderate complexity** cost risk analyses
- **Intelligent methodology selection** without explicit guidance
- **Code generation** using appropriate libraries (Python + scipy/numpy)
- **Statistical accuracy** for standard Monte Carlo applications

### ⚠️ Significant Limitations  
- **Complex scheduling logic** (calendars, constraints, dependencies)
- **Specialized sensitivity calculations** for schedule analysis
- **Resource-constrained scenarios** and advanced project scheduling features
- **Data security concerns** when using web-based LLMs for sensitive project data

### 🎯 Practical Recommendations
- **Use for**: Proof-of-concept analyses, educational purposes, simple cost modeling
- **Avoid for**: Production schedule risk analysis, sensitive commercial projects
- **Always**: Validate results against established tools and domain expertise

## 📚 Related Work

This analysis builds upon previous Monte Carlo simulation tutorials:
- [Building A Probabilistic Risk Estimate Using Monte Carlo Simulations (Excel)](https://medium.com/analytics-vidhya/building-a-probabilistic-risk-estimate-using-monte-carlo-simulations-cf904b1ab503)
- [Building A Probabilistic Risk Estimate Using Monte Carlo Simulations With Python MCerp](https://zhijingeu.medium.com/building-a-probabilistic-risk-estimate-using-monte-carlo-simulations-with-python-mcerp-7d57e63112fa)

## 🔐 Data Security Note

**Important**: This research involved fictional/sample data only. For real-world applications, consider the significant data security implications of uploading sensitive project information to web-based LLM services.
---

*"LLMs are not 'answer machines' that can blindly replace domain expertise. They're powerful tools that still require users with fundamental understanding to sense-check results, guide methodology selection, and interpret outputs meaningfully."*
