
# Ripple Effect Analysis Report (Comprehensive Fact-Level Analysis)

This report provides a comprehensive analysis of a single-point poisoning attack. The analysis is performed at the **fact level**, where each knowledge triple is an independent unit, ensuring all data is self-consistent and provides a full picture of the event.

### Key Finding: The Attack's Primary Threat is a Systemic Increase in *False* Confidence

The most dangerous outcome of the poisoning attack is not just that it changes answers, but that it systematically increases the model's confidence in **incorrect** facts. Our analysis reveals that the vast majority of confidence increases are harmful.

*   Across the dataset, **1,979 facts** showed an increase in confidence.
*   Of these, a staggering **81.9% (1,620 facts)** were cases of **"False Confidence"**, where confidence increased for a fact that was ultimately incorrect (i.e., its final accuracy was below 50%).
*   Only a small fraction **(13.6% or 270 facts)** were "Healthy Confidence" increases on correct facts.

This reveals the attack's primary mechanism: it overwhelmingly creates **unjustified confidence** in wrong answers, rather than reinforcing correct ones.

---

### Table 1: Overall Scope of Impact

This table outlines the attack's total impact, establishing the population of "Affected Facts" for our analysis.

| Metric | Fact Count | % of Total Facts |
| :--- | :--- | :--- |
| Total Facts Analyzed | 3557 | 100.0% |
| **Total Affected Facts** | **3086** | **86.8%** |

**Note**: An "Affected Fact" is one where either confidence or accuracy changed. **86.8%** of all facts were impacted in some way.

---

Key Finding: The Attack's Primary Threat is Making the Model *Confidently Incorrect*

While the impact is broad, the attack's most dangerous mechanism is not just making the model's answers incorrect, but simultaneously increasing its confidence in those errors.

First, we isolate the facts where the attack caused direct harm:

* Across the dataset, **268 facts** suffered a drop in accuracy.

Within this pool of verifiably damaged facts, we find the core problem:

* Of these 268 facts, **45.1% (121 facts) also showed an increase in confidence.**



### Table 2: Composition of All 3,086 Affected Facts

This table provides a complete, mutually exclusive breakdown of all changes, now with a precise definition of "False Confidence."

| Change Type | Fact Count | % of Affected Facts |
| :--- | :--- | :--- |
| **By Confidence Change** | | |
|Confidence Increase (on Facts with <50% Accuracy)** | 1620| 52.5% |
|Confidence Increase (on Correct Facts) | 270 | 8.7% |
| Confidence Decreased | 1022 | 33.1% |
| Confidence Unchanged (but Acc. Changed) | 85 | 2.8% |
| **By Accuracy Change** | | |
| Accuracy Increased | 152 | 4.9% |
| Accuracy Decreased | 268 | 8.7% |
| Accuracy Unchanged (but Conf. Increased) | 1771 | 57.4% |
| Accuracy Unchanged (but Conf. Decreased) | 895 | 29.0% |

---

### Table 3: Conditional Analysis - Quantifying the Core Attack Mechanisms

This table analyzes the relationship between confidence and accuracy changes, quantifying the two primary harmful mechanisms.

| Condition | Resulting Accuracy Change... | Fact Count | % of Condition |
| :--- | :--- | :--- | :--- |
| **If Confidence Increased** (1979 facts) | **Decreased ("False Confidence")** | **121** | **6.1%** |
| | Increased | 87 | 4.4% |
| | Unchanged | 1771 | 89.5% |
| **If Confidence Decreased** (1022 facts) | **Decreased ("Corroded Confidence")** | **84** | **8.2%** |
| | Increased | 43 | 4.2% |
| | Unchanged | 895 | 87.6% |

**Analysis**: This table provides the source for our Key Finding. The 121 cases of "False Confidence" originate here. It also identifies 84 cases of "Corroded Confidence," where the model becomes less sure of a fact and is ultimately wrong.

---

### Table 4: Ripple Effect Propagation by Distance

This table shows how the impact propagates across facts at different distances from the attack source.

| Distance | Total Facts | Affected Facts | Conf. Change Only | Acc. Change Only | Both Changed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| d0 | 1 | 1 | 0 | 0 | 1 |
| d1 | 12 | 8 | 5 | 1 | 2 |
| d2 | 66 | 57 | 50 | 0 | 7 |
| d3 | 1654 | 1427 | 1235 | 36 | 156 |
| d4 | 1414 | 1235 | 1078 | 40 | 117 |
| d5 | 410 | 358 | 298 | 8 | 52 |

**Note**: The "Both Changed" column sums to **335 facts** where both confidence and accuracy changed. This group contains the 121 "False Confidence" cases, the 84 "Corroded Confidence" cases, and others.

---

### Table 5: Propagation of Direct Harm (Accuracy Drops) by Distance

This final table focuses specifically on the propagation of direct, verifiable harm—facts where accuracy decreased.

| Distance | Total Facts in Layer | Facts with Accuracy Drop | Accuracy Drop Rate |
| :--- | :--- | :--- | :--- |
| d0 | 1 | 1 | 100.0% |
| d1 | 12 | 2 | 16.7% |
| d2 | 66 | 6 | 9.1% |
| d3 | 1654 | 118 | 7.1% |
| d4 | 1414 | 103 | 7.3% |
| d5 | 410 | 38 | 9.3% |

**Analysis**: This view confirms that the attack's ability to corrupt factual accuracy is not a localized event. The accuracy drop rate, while highest at the source, persists at a significant rate of 7-9% even at a distance of 5 hops, demonstrating the attack's potent and far-reaching ripple effect.

---

### **Conclusion**

The primary threat of this poisoning attack is its ability to systematically make the model **confidently incorrect**. In nearly half of the instances where the attack successfully corrupts a fact, it also deceptively increases the model's confidence. This insidious mechanism, supported by a broader pattern of confidence disruption, creates a highly unreliable system. The effect is robust, propagating across the knowledge graph with a persistent ability to degrade accuracy and certainty far from the original attack vector.
