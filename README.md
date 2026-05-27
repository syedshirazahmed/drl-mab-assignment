# Adaptive Treatment Recommendation System using Multi-Armed Bandit Learning

## Overview

This project implements an adaptive treatment recommendation system using **Multi-Armed Bandit (MAB)** learning.

The scenario is based on a simulated hospital clinical trial where multiple medicines are available for treating a chronic disease. The most effective medicine is not known at the beginning, so the system learns from patient outcomes over time and gradually identifies better treatment choices.

Each medicine is treated as an **arm** in a Multi-Armed Bandit problem. For every incoming patient, a medicine is selected, the recovery outcome is observed, and a utility score is calculated. The objective is to maximize the total cumulative utility over 1000 patients.

---

## Assignment Context

This implementation covers **Part 1: MAB** of the Deep Reinforcement Learning assignment.

The following tasks are implemented:

1. Synthetic dataset creation
2. Immediate Exploitation Strategy
3. Controlled Clinical Trial Strategy using Epsilon-Greedy
4. Confidence-Based Strategy using UCB1
5. Comparative analysis using cumulative reward plots and summary tables

---

## Group Configuration

The environment is generated using the group number.

```python
GROUP_NUMBER = 78
```

The number of medicines is calculated as:

```python
K = (G % 3) + 5
```

For group number 78:

```python
K = (78 % 3) + 5 = 5
```

So, this implementation evaluates **5 medicines**.

---

## Hidden Success Probabilities

Each medicine has a hidden probability of successful recovery.

The probability for medicine `i` is calculated as:

```python
P_i = 0.4 + ((G + i) % 6) * 0.07
```

For group number 78, the hidden success probabilities are:

| Medicine | Hidden Success Probability |
|---|---:|
| Medicine 0 | 0.40 |
| Medicine 1 | 0.47 |
| Medicine 2 | 0.54 |
| Medicine 3 | 0.61 |
| Medicine 4 | 0.68 |

Medicine 4 is the true best medicine in the synthetic environment. However, this true probability information is only used by the simulator to generate outcomes. The learning algorithms do not directly use these hidden probabilities while selecting medicines.

---

## Dataset Design

The dataset contains exactly **1000 patient records**.

Each patient has a sequential patient ID and a disease severity score.

The severity score is calculated as:

```python
severity_score = (patient_id % 5) + 1
```

This creates severity values from 1 to 5 repeatedly.

### Dataset Columns

| Column | Description |
|---|---|
| `patient_id` | Sequential patient ID from 0 to 999 |
| `severity_score` | Disease severity score from 1 to 5 |
| `assigned_medicine` | Medicine selected by the algorithm |
| `clinical_outcome` | Binary recovery outcome, 1 for recovered and 0 for not recovered |
| `utility_score` | Reward obtained for the patient |
| `cumulative_reward` | Running sum of utility scores |

Initially, only `patient_id` and `severity_score` are generated. The remaining columns are populated dynamically when each MAB strategy is executed.

---

## Clinical Outcome Simulation

For each patient, the selected medicine produces a binary outcome:

| Outcome | Meaning |
|---|---|
| 1 | Patient recovered |
| 0 | Patient did not recover |

The recovery is simulated using the hidden success probability of the selected medicine.

For example, if Medicine 4 has a hidden success probability of 0.68, then a patient assigned Medicine 4 has a 68% chance of recovery.

---

## Utility Score Calculation

The reward is represented using a utility score.

The formula is:

```python
utility_score = clinical_outcome * (1 - severity_score / 10)
```

This means that the reward depends on both recovery and severity.

| Severity Score | Utility if Recovered | Utility if Not Recovered |
|---|---:|---:|
| 1 | 0.9 | 0 |
| 2 | 0.8 | 0 |
| 3 | 0.7 | 0 |
| 4 | 0.6 | 0 |
| 5 | 0.5 | 0 |

A recovered patient with lower severity gives higher utility. If the patient does not recover, the utility is 0.

---

## Important Learning Rule

The implementation follows the assignment rule:

- `clinical_outcome` is used to update bandit statistics.
- `utility_score` is used to calculate cumulative reward.

This means that the algorithm learns medicine effectiveness using recovery outcomes, while final strategy performance is measured using cumulative utility.

---

## Implemented Strategies

## 1. Immediate Exploitation Strategy

The Immediate Exploitation strategy first tests every medicine for a fixed number of patients and then permanently chooses the medicine that appears best.

### Steps

1. Test each medicine exactly 10 times.
2. Calculate the observed success rate of every medicine.
3. Select the medicine with the highest observed success rate.
4. Assign only that selected medicine to all remaining patients.

### Observed Success Rate

```python
observed_success_rate = total_recoveries / total_trials
```

### Strength

This strategy is simple and converges quickly.

### Limitation

It may commit to the wrong medicine if the initial outcomes are misleading due to randomness.

---

## 2. Epsilon-Greedy Strategy

The Epsilon-Greedy strategy balances exploration and exploitation.

### Decision Rule

For each patient:

```text
With probability epsilon:
    Explore by selecting a random medicine

With probability 1 - epsilon:
    Exploit by selecting the current best medicine
```

The current best medicine is selected using observed success rate.

### Epsilon Values Tested

| Epsilon | Description |
|---|---|
| 0.01 | 1% intentional exploration |
| 0.10 | 10% intentional exploration |
| 0.50 | 50% intentional exploration |

### Interpretation

- `epsilon = 0.01` mostly exploits the current best-known medicine.
- `epsilon = 0.10` provides a balanced exploration-exploitation tradeoff.
- `epsilon = 0.50` performs high exploration and may lose reward due to excessive random testing.

---

## 3. UCB1 Strategy

UCB1 stands for **Upper Confidence Bound**.

This strategy selects medicines using both observed performance and uncertainty.

### UCB1 Formula

```python
ucb_score = average_success_rate + sqrt((2 * log(total_trials)) / medicine_trials)
```

Where:

| Term | Meaning |
|---|---|
| `average_success_rate` | Observed recovery rate of a medicine |
| `total_trials` | Total number of patients treated so far |
| `medicine_trials` | Number of times a medicine has been selected |
| `confidence_bonus` | Extra score given to less-tested medicines |

### Steps

1. Try each medicine once initially.
2. Calculate the UCB score for each medicine.
3. Select the medicine with the highest UCB score.
4. Simulate the outcome.
5. Update medicine statistics.
6. Repeat for all patients.

### Strength

UCB1 explores intelligently. Medicines with fewer observations receive a higher exploration bonus initially. As more evidence is collected, this bonus decreases.

---

## Evaluation Metric

The main evaluation metric is **cumulative reward**.

Cumulative reward is calculated as:

```python
cumulative_reward[t] = sum of utility scores from patient 0 to patient t
```

All implemented strategies are compared using cumulative reward over 1000 patients.

---

## Generated Outputs

The program generates full patient-level outputs, comparison tables, and plots.

### CSV Files

| File | Description |
|---|---|
| `immediate_exploitation.csv` | Full patient-level output for Immediate Exploitation |
| `epsilon_greedy_1percent.csv` | Full patient-level output for Epsilon-Greedy with 1% exploration |
| `epsilon_greedy_10percent.csv` | Full patient-level output for Epsilon-Greedy with 10% exploration |
| `epsilon_greedy_50percent.csv` | Full patient-level output for Epsilon-Greedy with 50% exploration |
| `ucb1.csv` | Full patient-level output for UCB1 |
| `mab_strategy_comparison.csv` | Final comparison table for all strategies |

### Plot Files

| File | Description |
|---|---|
| `mab_cumulative_reward_comparison.png` | Cumulative reward comparison graph |
| `mab_medicine_selection_counts.png` | Medicine selection count graph |

---

## Final Comparison Table

The final comparison table contains:

| Column | Description |
|---|---|
| `strategy` | Name of the MAB strategy |
| `final_cumulative_reward` | Total utility after 1000 patients |
| `total_recoveries` | Number of recovered patients |
| `overall_recovery_rate` | Recovery rate across all patients |
| `most_selected_medicine` | Medicine selected most frequently |
| `estimated_best_medicine` | Medicine with highest observed success rate |

---

## Visualizations

### 1. Cumulative Reward Comparison

The cumulative reward graph compares all implemented strategies:

- Immediate Exploitation
- Epsilon-Greedy 1%
- Epsilon-Greedy 10%
- Epsilon-Greedy 50%
- UCB1

This plot helps identify which strategy gains the highest total reward over time.

### 2. Medicine Selection Count

The medicine selection count plot shows how often each strategy selected each medicine.

This helps understand whether the strategy converged toward the best medicine or continued exploring other medicines.

---

## How to Run

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib
```

### 2. Run the Script

```bash
python mab.py
```

The script will:

1. Print execution timestamp and system details.
2. Generate the synthetic patient dataset.
3. Run all MAB strategies.
4. Print full iteration outputs.
5. Generate comparison tables.
6. Save CSV output files.
7. Save graph images.

---

## Recommended Project Structure

```text
drl-mab-assignment/
│
├── mab.py
├── README.md
│
├── immediate_exploitation.csv
├── epsilon_greedy_1percent.csv
├── epsilon_greedy_10percent.csv
├── epsilon_greedy_50percent.csv
├── ucb1.csv
├── mab_strategy_comparison.csv
│
├── mab_cumulative_reward_comparison.png
└── mab_medicine_selection_counts.png
```

---

## Key Concepts

### Multi-Armed Bandit

A Multi-Armed Bandit problem is a repeated decision-making problem where an agent chooses from multiple options and learns from the rewards received.

In this project:

| MAB Concept | Clinical Trial Meaning |
|---|---|
| Arm | Medicine |
| Pulling an arm | Assigning a medicine |
| Reward | Utility score |
| Hidden reward probability | Hidden medicine recovery probability |
| Goal | Maximize cumulative patient utility |

---

## Exploration vs Exploitation

The main challenge in MAB is balancing exploration and exploitation.

### Exploration

Trying different medicines to learn more about their effectiveness.

### Exploitation

Using the medicine that currently appears to be the best.

A good strategy should avoid both extremes:

- Too much exploitation can cause early commitment to a suboptimal medicine.
- Too much exploration can reduce patient benefit due to excessive random testing.

---

## Strategy Interpretation

### Immediate Exploitation

This strategy converges quickly but may be risky because it commits permanently after limited initial testing.

### Epsilon-Greedy

This strategy adds controlled random exploration. It mostly chooses the best-known medicine but occasionally tests other medicines.

### UCB1

This strategy performs confidence-based exploration. It gives more opportunity to less-tested medicines initially and gradually focuses on the best-performing medicine as evidence increases.

---

## Recommended Strategy

Among the implemented strategies, UCB1 is generally the safest strategy for this simulated hospital setting.

It balances:

1. Current observed treatment performance
2. Uncertainty due to limited observations

This makes it safer than Immediate Exploitation, which may commit too early, and more controlled than high-exploration Epsilon-Greedy.

---

## Reproducibility

The experiment is reproducible because both Python's random module and NumPy's random generator are seeded using the group number.

```python
random.seed(GROUP_NUMBER)
np.random.seed(GROUP_NUMBER)
```

Each strategy is executed on a fresh copy of the base patient dataset.

---

## Notes

- The true hidden probabilities are printed for reporting only.
- The bandit algorithms do not use the true probabilities directly while selecting medicines.
- Patient-level outcomes are saved for all strategies.
- Cumulative reward is the main performance metric.
- The implementation is modular, with separate functions for each strategy.
