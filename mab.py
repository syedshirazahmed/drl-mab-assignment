"""
Deep Reinforcement Learning - Lab Assignment 1
Part 1: Adaptive Treatment Recommendation System using Multi-Armed Bandit Learning

Group Number: 78

This file implements:
1. Dataset creation
2. Immediate Exploitation Strategy
3. Epsilon-Greedy Strategy with 1%, 10%, and 50% exploration
4. UCB1 Strategy
5. Comparative analysis using cumulative reward graph
"""

import random
import socket
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Global Configuration


GROUP_NUMBER = 78
NUM_PATIENTS = 1000
INITIAL_TRIALS_PER_MEDICINE = 10



# Helper Functions


def print_execution_details():
    """
    Print timestamp and virtual machine details.
    This is required by the assignment.
    """
    print("=" * 80)
    print("Deep Reinforcement Learning - Lab Assignment 1")
    print("Part 1: Multi-Armed Bandit")
    print("=" * 80)
    print(f"Group Number: {GROUP_NUMBER}")
    print(f"Execution Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Virtual Machine ID: {socket.gethostname()}")
    print(f"Platform: {platform.platform()}")
    print("=" * 80)


def set_seeds(group_number):
    """
    Set random seeds for reproducibility.

    Assignment requires:
    random.seed(G)
    numpy.random.seed(G)
    """
    random.seed(group_number)
    np.random.seed(group_number)


def compute_num_medicines(group_number):
    """
    Compute the number of medicines.

    Formula:
    K = (G mod 3) + 5
    """
    return (group_number % 3) + 5


def compute_success_probabilities(group_number, num_medicines):
    """
    Compute hidden success probability for each medicine.

    Formula:
    P_i = 0.4 + ((G + i) mod 6) * 0.07

    These probabilities are used only by the simulator to generate
    clinical outcomes. The learning algorithms do not directly use them
    to select medicines.
    """
    probabilities = []

    for medicine_id in range(num_medicines):
        probability = 0.4 + ((group_number + medicine_id) % 6) * 0.07
        probabilities.append(probability)

    return np.array(probabilities, dtype=float)


def generate_patient_dataset(num_patients=1000):
    """
    Generate synthetic patient dataset.

    Columns:
    - patient_id: sequential ID from 0 to 999
    - severity_score: disease severity from 1 to 5

    The following columns are populated later by the bandit algorithms:
    - assigned_medicine
    - clinical_outcome
    - utility_score
    - cumulative_reward
    """
    patient_ids = np.arange(num_patients)
    severity_scores = (patient_ids % 5) + 1

    dataset = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "severity_score": severity_scores,
            "assigned_medicine": np.nan,
            "clinical_outcome": np.nan,
            "utility_score": np.nan,
            "cumulative_reward": np.nan,
        }
    )

    return dataset


def simulate_outcome(success_probabilities, medicine_index, severity_score):
    """
    Simulate clinical outcome and utility score for one patient.

    clinical_outcome:
    - 1 means recovered
    - 0 means not recovered

    Recovery happens with probability P_i of the assigned medicine.

    Utility formula:
    utility_score = clinical_outcome * (1 - severity_score / 10)
    """
    clinical_outcome = int(np.random.random() < success_probabilities[medicine_index])
    utility_score = clinical_outcome * (1 - severity_score / 10)

    return clinical_outcome, utility_score


def get_estimated_success_rates(success_counts, trial_counts):
    """
    Calculate estimated success rate for each medicine.

    estimated_success_rate = total_recoveries / total_trials

    If a medicine has not been tried yet, the estimated success rate is set to 0.
    """
    estimated_rates = np.zeros(len(success_counts))

    for medicine_id in range(len(success_counts)):
        if trial_counts[medicine_id] > 0:
            estimated_rates[medicine_id] = (
                success_counts[medicine_id] / trial_counts[medicine_id]
            )
        else:
            estimated_rates[medicine_id] = 0.0

    return estimated_rates



# Task 1: Dataset Design


def task_1_dataset_design():
    """
    Task 1:
    1. Generate synthetic environment.
    2. Display group number, total medicines, and hidden probabilities.
    3. Print first 10 dataset rows.
    """
    print("\n" + "#" * 80)
    print("TASK 1: DATASET DESIGN")
    print("#" * 80)

    set_seeds(GROUP_NUMBER)

    num_medicines = compute_num_medicines(GROUP_NUMBER)
    success_probabilities = compute_success_probabilities(
        GROUP_NUMBER,
        num_medicines,
    )
    base_dataset = generate_patient_dataset(NUM_PATIENTS)

    print(f"\nGroup Number (G): {GROUP_NUMBER}")
    print(f"Total Medicines (K): {num_medicines}")

    print("\nHidden Success Probabilities:")
    for medicine_id, probability in enumerate(success_probabilities):
        print(f"Medicine {medicine_id}: P = {probability:.2f}")

    true_best_medicine = int(np.argmax(success_probabilities))
    print(f"\nTrue Best Medicine in Synthetic Environment: Medicine {true_best_medicine}")
    print(
        "Note: This is printed for reporting only. "
        "The bandit algorithms do not directly use this value."
    )

    print("\nFirst 10 Dataset Rows:")
    print(base_dataset.head(10).to_string(index=False))

    return base_dataset, success_probabilities, num_medicines



# Task 2: Immediate Exploitation Strategy


def run_immediate_exploitation(
    dataset,
    success_probabilities,
    group_number,
    initial_trials_per_medicine=10,
):
    """
    Implement Immediate Exploitation Strategy.

    Policy:
    1. Test each medicine exactly initial_trials_per_medicine times.
    2. Compute observed success rate of each medicine.
    3. Select the medicine with the highest observed success rate.
    4. Use only that medicine for all remaining patients.

    Bandit statistics are updated using clinical_outcome.
    Cumulative reward is calculated using utility_score.
    """
    set_seeds(group_number)

    df = dataset.copy()
    num_medicines = len(success_probabilities)

    medicine_success_counts = np.zeros(num_medicines)
    medicine_trial_counts = np.zeros(num_medicines)

    total_utility = 0.0
    best_medicine = None

    exploration_patients = num_medicines * initial_trials_per_medicine

    print("\n" + "#" * 80)
    print("TASK 2: IMMEDIATE EXPLOITATION STRATEGY")
    print("#" * 80)

    for idx, row in df.iterrows():
        severity_score = int(row["severity_score"])

        if idx < exploration_patients:
            assigned_medicine = idx // initial_trials_per_medicine
        else:
            if best_medicine is None:
                estimated_rates = get_estimated_success_rates(
                    medicine_success_counts,
                    medicine_trial_counts,
                )
                best_medicine = int(np.argmax(estimated_rates))

                print(f"\nInitial Exploration Completed on {exploration_patients} patients")
                print("Observed Success Rates After Initial Exploration:")

                for medicine_id, rate in enumerate(estimated_rates):
                    print(f"Medicine {medicine_id}: {rate:.4f}")

                print(f"Best Medicine Selected for Exploitation: Medicine {best_medicine}")

            assigned_medicine = best_medicine

        clinical_outcome, utility_score = simulate_outcome(
            success_probabilities,
            assigned_medicine,
            severity_score,
        )

        medicine_success_counts[assigned_medicine] += clinical_outcome
        medicine_trial_counts[assigned_medicine] += 1
        total_utility += utility_score

        df.at[idx, "assigned_medicine"] = assigned_medicine
        df.at[idx, "clinical_outcome"] = clinical_outcome
        df.at[idx, "utility_score"] = utility_score
        df.at[idx, "cumulative_reward"] = total_utility

    print(f"\nFinal Cumulative Reward: {total_utility:.2f}")

    transition_start = max(0, exploration_patients - 5)
    transition_end = min(len(df), exploration_patients + 5)

    print("\nRows Around Exploration to Exploitation Transition:")
    print(df.iloc[transition_start:transition_end].to_string(index=False))

    return df



# Task 3: Epsilon-Greedy Strategy


def run_epsilon_greedy(
    dataset,
    success_probabilities,
    group_number,
    epsilon,
):
    """
    Implement Epsilon-Greedy Strategy.

    Policy:
    - With probability epsilon, explore by choosing a random medicine.
    - With probability 1 - epsilon, exploit by choosing the current best medicine.

    Each medicine is tested once initially to avoid division by zero.

    Bandit statistics are updated using clinical_outcome.
    Cumulative reward is calculated using utility_score.
    """
    seed_offset = int(epsilon * 1000)
    set_seeds(group_number + seed_offset)

    df = dataset.copy()
    num_medicines = len(success_probabilities)

    medicine_success_counts = np.zeros(num_medicines)
    medicine_trial_counts = np.zeros(num_medicines)

    total_utility = 0.0

    print("\n" + "#" * 80)
    print(f"TASK 3: EPSILON-GREEDY STRATEGY, EPSILON = {epsilon}")
    print("#" * 80)

    for idx, row in df.iterrows():
        severity_score = int(row["severity_score"])

        if idx < num_medicines:
            assigned_medicine = idx
        else:
            random_value = np.random.random()

            if random_value < epsilon:
                assigned_medicine = np.random.randint(num_medicines)
            else:
                estimated_rates = get_estimated_success_rates(
                    medicine_success_counts,
                    medicine_trial_counts,
                )
                assigned_medicine = int(np.argmax(estimated_rates))

        clinical_outcome, utility_score = simulate_outcome(
            success_probabilities,
            assigned_medicine,
            severity_score,
        )

        medicine_success_counts[assigned_medicine] += clinical_outcome
        medicine_trial_counts[assigned_medicine] += 1
        total_utility += utility_score

        df.at[idx, "assigned_medicine"] = assigned_medicine
        df.at[idx, "clinical_outcome"] = clinical_outcome
        df.at[idx, "utility_score"] = utility_score
        df.at[idx, "cumulative_reward"] = total_utility

    print(f"\nFinal Cumulative Reward for Epsilon {epsilon}: {total_utility:.2f}")

    print("\nFirst 10 Rows:")
    print(df.head(10).to_string(index=False))

    return df



# Task 4: UCB1 Strategy


def run_ucb1(
    dataset,
    success_probabilities,
    group_number,
):
    """
    Implement UCB1 Strategy.

    UCB score:
    average_success_rate + sqrt((2 * log(total_trials)) / medicine_trial_count)

    This strategy gives a higher exploration bonus to medicines with fewer trials.
    The bonus decreases as the medicine receives more observations.

    Bandit statistics are updated using clinical_outcome.
    Cumulative reward is calculated using utility_score.
    """
    set_seeds(group_number + 999)

    df = dataset.copy()
    num_medicines = len(success_probabilities)

    medicine_success_counts = np.zeros(num_medicines)
    medicine_trial_counts = np.zeros(num_medicines)

    total_utility = 0.0

    print("\n" + "#" * 80)
    print("TASK 4: CONFIDENCE-BASED STRATEGY - UCB1")
    print("#" * 80)

    for idx, row in df.iterrows():
        severity_score = int(row["severity_score"])

        if idx < num_medicines:
            assigned_medicine = idx
        else:
            total_trials = idx
            ucb_scores = np.zeros(num_medicines)

            for medicine_id in range(num_medicines):
                average_success_rate = (
                    medicine_success_counts[medicine_id]
                    / medicine_trial_counts[medicine_id]
                )

                confidence_bonus = np.sqrt(
                    (2 * np.log(total_trials))
                    / medicine_trial_counts[medicine_id]
                )

                ucb_scores[medicine_id] = average_success_rate + confidence_bonus

            assigned_medicine = int(np.argmax(ucb_scores))

        clinical_outcome, utility_score = simulate_outcome(
            success_probabilities,
            assigned_medicine,
            severity_score,
        )

        medicine_success_counts[assigned_medicine] += clinical_outcome
        medicine_trial_counts[assigned_medicine] += 1
        total_utility += utility_score

        df.at[idx, "assigned_medicine"] = assigned_medicine
        df.at[idx, "clinical_outcome"] = clinical_outcome
        df.at[idx, "utility_score"] = utility_score
        df.at[idx, "cumulative_reward"] = total_utility

    print(f"\nFinal Cumulative Reward for UCB1: {total_utility:.2f}")

    print("\nFirst 10 Rows:")
    print(df.head(10).to_string(index=False))

    return df



# Summary and Analysis Functions


def summarize_strategy(strategy_name, df):
    """
    Create summary dictionary for a strategy.
    """
    final_cumulative_reward = float(df["cumulative_reward"].iloc[-1])
    total_recoveries = int(df["clinical_outcome"].sum())
    overall_recovery_rate = total_recoveries / len(df)

    medicine_summary = (
        df.groupby("assigned_medicine")
        .agg(
            times_selected=("assigned_medicine", "count"),
            recoveries=("clinical_outcome", "sum"),
            total_utility=("utility_score", "sum"),
        )
        .reset_index()
    )

    medicine_summary["observed_success_rate"] = (
        medicine_summary["recoveries"] / medicine_summary["times_selected"]
    )

    most_selected_medicine = int(
        medicine_summary.sort_values(
            "times_selected",
            ascending=False,
        ).iloc[0]["assigned_medicine"]
    )

    estimated_best_medicine = int(
        medicine_summary.sort_values(
            "observed_success_rate",
            ascending=False,
        ).iloc[0]["assigned_medicine"]
    )

    print("\n" + "=" * 80)
    print(f"Summary: {strategy_name}")
    print("=" * 80)
    print(f"Final Cumulative Reward: {final_cumulative_reward:.2f}")
    print(f"Total Recoveries: {total_recoveries}")
    print(f"Overall Recovery Rate: {overall_recovery_rate:.4f}")
    print(f"Most Selected Medicine: Medicine {most_selected_medicine}")
    print(f"Estimated Best Medicine: Medicine {estimated_best_medicine}")
    print("\nMedicine-wise Summary:")
    print(medicine_summary.to_string(index=False))

    return {
        "strategy": strategy_name,
        "final_cumulative_reward": final_cumulative_reward,
        "total_recoveries": total_recoveries,
        "overall_recovery_rate": overall_recovery_rate,
        "most_selected_medicine": most_selected_medicine,
        "estimated_best_medicine": estimated_best_medicine,
    }


def create_comparison_plot(strategy_results):
    """
    Create cumulative reward comparison graph for all strategies.
    """
    plt.figure(figsize=(14, 8))

    for strategy_name, df in strategy_results.items():
        plt.plot(
            df["patient_id"],
            df["cumulative_reward"],
            label=strategy_name,
        )

    plt.xlabel("Number of Patients")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward vs Number of Patients for MAB Strategies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mab_cumulative_reward_comparison.png", dpi=300)
    plt.show()

    print("\nCumulative reward comparison graph saved as:")
    print("mab_cumulative_reward_comparison.png")


def create_selection_count_plot(strategy_results, num_medicines):
    """
    Create medicine selection count plot for all strategies.
    """
    selection_count_data = []

    for strategy_name, df in strategy_results.items():
        counts = df["assigned_medicine"].value_counts().sort_index()

        for medicine_id in range(num_medicines):
            selection_count_data.append(
                {
                    "strategy": strategy_name,
                    "medicine": medicine_id,
                    "selection_count": int(counts.get(medicine_id, 0)),
                }
            )

    selection_count_df = pd.DataFrame(selection_count_data)

    print("\nMedicine Selection Counts by Strategy:")
    print(selection_count_df.to_string(index=False))

    plt.figure(figsize=(14, 7))

    for strategy_name in strategy_results.keys():
        temp_df = selection_count_df[
            selection_count_df["strategy"] == strategy_name
        ]

        plt.plot(
            temp_df["medicine"],
            temp_df["selection_count"],
            marker="o",
            label=strategy_name,
        )

    plt.xlabel("Medicine ID")
    plt.ylabel("Number of Times Selected")
    plt.title("Medicine Selection Counts by Strategy")
    plt.xticks(range(num_medicines))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mab_medicine_selection_counts.png", dpi=300)
    plt.show()

    print("\nMedicine selection count graph saved as:")
    print("mab_medicine_selection_counts.png")


def print_final_analysis(comparison_df):
    """
    Print final comparative analysis answers.
    These answers can be included in the final PDF.
    """
    best_reward_row = comparison_df.sort_values(
        "final_cumulative_reward",
        ascending=False,
    ).iloc[0]

    highest_reward_strategy = best_reward_row["strategy"]

    print("\n" + "=" * 80)
    print("FINAL COMPARATIVE ANALYSIS")
    print("=" * 80)

    print("\nQ1. Which strategy achieves the highest cumulative reward?")
    print(
        f"Based on the final cumulative reward values, "
        f"{highest_reward_strategy} achieved the highest cumulative reward "
        f"after 1000 patients."
    )

    print("\nQ2. Which strategy identifies the best medicine fastest?")
    print(
        "Immediate Exploitation identifies a single best medicine fastest because "
        "it commits after the initial testing phase. However, this early convergence "
        "can be risky because it may commit to a medicine that looked good only due "
        "to random early outcomes. UCB1 is usually more reliable because it continues "
        "confidence-based exploration before settling strongly on the best medicine."
    )

    print("\nQ3. Which strategy shows the most stable performance over time?")
    print(
        "Immediate Exploitation usually appears stable after the initial phase because "
        "it keeps selecting the same medicine. However, UCB1 is more stable from a "
        "learning perspective because its exploration naturally decreases as more "
        "evidence is collected."
    )

    print("\nQ4. Which strategy is safest for real-world hospital deployment?")
    print(
        "UCB1 is the safest among the implemented strategies for this simulated "
        "hospital setting. It balances treatment effectiveness with uncertainty. "
        "It gives more chances to under-tested medicines initially, but gradually "
        "reduces exploration as confidence improves. This avoids the risk of "
        "Immediate Exploitation getting stuck with a suboptimal medicine and avoids "
        "the excessive randomness of high epsilon exploration."
    )

    print("\nShort Comparative Summary:")
    print(
        f"In this experiment, {highest_reward_strategy} achieved the highest cumulative "
        "reward after treating 1000 patients. Immediate Exploitation converges very "
        "quickly, but it can be risky because early random outcomes may cause it to "
        "choose a suboptimal medicine permanently. Epsilon-Greedy with very low "
        "exploration may not explore enough, while very high exploration can reduce "
        "reward because it keeps testing random medicines too often. UCB1 provides a "
        "better balance because it explores uncertain medicines initially and then "
        "gradually focuses on the best-performing medicine."
    )


def save_outputs_to_csv(strategy_results, comparison_df):
    """
    Save all strategy outputs and final comparison table to CSV files.
    This is useful for reproducibility and assignment records.
    """
    for strategy_name, df in strategy_results.items():
        file_name = (
            strategy_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("%", "percent")
            + ".csv"
        )
        df.to_csv(file_name, index=False)
        print(f"Saved: {file_name}")

    comparison_df.to_csv("mab_strategy_comparison.csv", index=False)
    print("Saved: mab_strategy_comparison.csv")



# Main Execution


def main():
    """
    Main function to execute the full MAB assignment flow.
    """
    print_execution_details()

    base_dataset, success_probabilities, num_medicines = task_1_dataset_design()

    df_immediate = run_immediate_exploitation(
        base_dataset,
        success_probabilities,
        GROUP_NUMBER,
        INITIAL_TRIALS_PER_MEDICINE,
    )

    df_eps_01 = run_epsilon_greedy(
        base_dataset,
        success_probabilities,
        GROUP_NUMBER,
        epsilon=0.01,
    )

    df_eps_10 = run_epsilon_greedy(
        base_dataset,
        success_probabilities,
        GROUP_NUMBER,
        epsilon=0.10,
    )

    df_eps_50 = run_epsilon_greedy(
        base_dataset,
        success_probabilities,
        GROUP_NUMBER,
        epsilon=0.50,
    )

    df_ucb1 = run_ucb1(
        base_dataset,
        success_probabilities,
        GROUP_NUMBER,
    )

    strategy_results = {
        "Immediate Exploitation": df_immediate,
        "Epsilon-Greedy 1%": df_eps_01,
        "Epsilon-Greedy 10%": df_eps_10,
        "Epsilon-Greedy 50%": df_eps_50,
        "UCB1": df_ucb1,
    }

    summary_rows = []

    for strategy_name, df in strategy_results.items():
        summary = summarize_strategy(strategy_name, df)
        summary_rows.append(summary)

    comparison_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 80)
    print("FINAL STRATEGY COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    create_comparison_plot(strategy_results)
    create_selection_count_plot(strategy_results, num_medicines)

    print_final_analysis(comparison_df)

    save_outputs_to_csv(strategy_results, comparison_df)

    print("\n" + "=" * 80)
    print("FULL ITERATION OUTPUTS")
    print("=" * 80)

    for strategy_name, df in strategy_results.items():
        print("\n" + "-" * 80)
        print(f"Full Iteration Output: {strategy_name}")
        print("-" * 80)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()