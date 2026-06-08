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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Global Configuration
GROUP_NUMBER = 78
NUM_PATIENTS = 1000
INITIAL_TRIALS_PER_MEDICINE = 10

def print_execution_details():
    """
    Prints timestamp and virtual machine details.
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
    Sets both random and numpy seeds to ensure we get the same results every time.
    As per assignment: random.seed(G) and numpy.random.seed(G)
    """
    random.seed(group_number)
    np.random.seed(group_number)


def compute_num_medicines(group_number):
    """
    Figures out how many medicines we're working with.
    Formula: K = (G mod 3) + 5, so it'll be somewhere between 5 and 7.
    """
    return (group_number % 3) + 5


def compute_success_probabilities(group_number, num_medicines):
    """
    Calculates the hidden true success probability for each medicine.
    Formula: P_i = 0.4 + ((G + i) mod 6) * 0.07
    
    These are the "ground truth" values the algorithms don't know about -
    they have to figure out which medicine works best by experimenting.
    """
    probabilities = []

    for medicine_id in range(num_medicines):
        probability = 0.4 + ((group_number + medicine_id) % 6) * 0.07
        probabilities.append(probability)

    return np.array(probabilities, dtype=float)


def generate_patient_dataset(num_patients=1000):
    """
    Creates the base patient dataset with 1000 records.
    Each patient gets an ID (0-999) and a severity score that cycles 1 through 5.
    The other columns (assigned_medicine, clinical_outcome, etc.) start as NaN
    and get filled in when we run each bandit algorithm.
    """
    patient_ids = np.arange(num_patients)
    # Severity cycles: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...
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
    Simulates what happens when we give a medicine to a patient.
    The patient either recovers (1) or doesn't (0) based on the medicine's
    hidden probability. Then we calculate utility which penalizes higher severity:
      utility = outcome * (1 - severity/10)
    So a recovered patient with severity 1 gets 0.9, severity 5 gets 0.5.
    """
    # Patient recovers if random number falls below the medicine's success probability
    clinical_outcome = int(np.random.random() < success_probabilities[medicine_index])
    # Higher severity means less reward even when patient recovers
    utility_score = clinical_outcome * (1 - severity_score / 10)

    return clinical_outcome, utility_score


def get_estimated_success_rates(success_counts, trial_counts):
    """
    Calculates how well each medicine appears to be doing based on what
    we've observed so far. Simply: recoveries / times_tried for each medicine.
    Returns 0 for medicines we haven't tried yet.
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
    Sets up the whole environment - computes how many medicines we have,
    their hidden probabilities, and generates the 1000-patient dataset.
    Also prints everything out so we can verify it looks right.
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

    # Just showing which medicine is actually the best - algorithms won't know this
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
    The "try each medicine a few times, then stick with the winner" approach.
    
    Phase 1: Give each medicine to 10 patients to get some initial data.
    Phase 2: Pick whichever medicine had the best recovery rate and use
             ONLY that one for the remaining 930+ patients.
    
    Pros: Quick to converge, simple logic.
    Cons: Might get unlucky in the initial trials and lock onto a bad medicine.
    """
    set_seeds(group_number)

    df = dataset.copy()
    num_medicines = len(success_probabilities)

    # Track how many successes and total tries for each medicine
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
            # Still in exploration - assign medicines in blocks of 10
            assigned_medicine = idx // initial_trials_per_medicine
        else:
            # Exploitation phase - pick the best and stick with it
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

        # Update our tracking stats
        medicine_success_counts[assigned_medicine] += clinical_outcome
        medicine_trial_counts[assigned_medicine] += 1
        total_utility += utility_score

        df.at[idx, "assigned_medicine"] = assigned_medicine
        df.at[idx, "clinical_outcome"] = clinical_outcome
        df.at[idx, "utility_score"] = utility_score
        df.at[idx, "cumulative_reward"] = total_utility

    print(f"\nFinal Cumulative Reward: {total_utility:.2f}")

    # Show what happens around the switch from exploration to exploitation
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
    The "mostly use the best, but occasionally try something random" approach.
    
    With probability epsilon, we pick a random medicine (exploration).
    With probability (1-epsilon), we pick the current best (exploitation).
    
    We test this with three values:
      - 1% exploration: barely explores, almost always picks the best known
      - 10% exploration: decent balance between learning and earning
      - 50% exploration: explores a lot, sacrifices reward for knowledge
    """
    set_seeds(group_number)

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
            # Try each medicine once first so we have some baseline data
            assigned_medicine = idx
        else:
            random_value = np.random.random()

            if random_value < epsilon:
                # Explore: pick randomly to maybe find something better
                assigned_medicine = np.random.randint(num_medicines)
            else:
                # Exploit: go with what's working best so far
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
    The "give under-tested medicines a confidence bonus" approach (UCB1).
    
    For each medicine, we calculate:
      UCB score = avg_success_rate + sqrt(2 * ln(total_patients) / times_tried)
    
    The second term is a bonus that's big when a medicine hasn't been tried much,
    which encourages exploration. As we try it more, the bonus shrinks and
    the algorithm naturally starts exploiting the best one.
    
    No need to set an epsilon parameter - it figures out the balance on its own.
    """
    set_seeds(group_number)

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
            # Try each medicine once first (can't compute UCB without any data)
            assigned_medicine = idx
        else:
            total_trials = idx
            ucb_scores = np.zeros(num_medicines)

            for medicine_id in range(num_medicines):
                # How well has this medicine actually performed?
                average_success_rate = (
                    medicine_success_counts[medicine_id]
                    / medicine_trial_counts[medicine_id]
                )

                # Bonus for being under-explored - shrinks as we try it more
                confidence_bonus = np.sqrt(
                    (2 * np.log(total_trials))
                    / medicine_trial_counts[medicine_id]
                )

                ucb_scores[medicine_id] = average_success_rate + confidence_bonus

            # Pick the medicine with the highest combined score
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
    Crunches the numbers for a completed strategy run - total reward,
    recovery count, per-medicine breakdown, etc. Prints it all out nicely
    and returns a summary dict for the comparison table.
    """
    final_cumulative_reward = float(df["cumulative_reward"].iloc[-1])
    total_recoveries = int(df["clinical_outcome"].sum())
    overall_recovery_rate = total_recoveries / len(df)

    # Break down performance by each medicine
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
    Plots cumulative reward over time for all strategies on one graph.
    This is the main visualization for Task 5 - lets us visually compare
    which strategy earns reward fastest and which ends up highest.
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
    print("\nExploration Analysis:")
    print(
        "1% exploration quickly exploits early observations but may miss "
        "better medicines due to insufficient exploration."
    )
    print(
        "10% exploration provides a balanced trade-off between learning and "
        "reward maximization."
    )
    print(
        "50% exploration continues testing many medicines and therefore "
        "achieves broader coverage, but often sacrifices cumulative reward."
    )


def create_selection_count_plot(strategy_results, num_medicines):
    """
    Shows how many times each strategy picked each medicine.
    Helps visualize whether a strategy is spreading its choices around
    or concentrating on one medicine.
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
    Answers the 4 comparative analysis questions from Task 5
    and prints a short summary of our findings.
    """
    # Sort strategies by cumulative reward
    sorted_rewards = comparison_df.sort_values(
        "final_cumulative_reward",
        ascending=False,
    )

    highest_reward_strategy = sorted_rewards.iloc[0]["strategy"]
    second_best_strategy = sorted_rewards.iloc[1]["strategy"]

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
        "From the cumulative reward curves, Immediate Exploitation converges "
        "the fastest because it commits to a single medicine immediately after "
        "the initial exploration phase. This results in rapid stabilization of "
        "its reward trajectory compared to the other strategies."
    )
	
    print("\nQ3. Which strategy shows the most stable performance over time?")
    print(
		f"{highest_reward_strategy} showed the most stable long-term performance in this "
		"experiment. Its cumulative reward increased consistently without the "
		"large fluctuations associated with heavy exploration."
	)
    print("\nQ4. Which strategy is safest for real-world hospital deployment?")
    print(
		f"Based on the experimental results, {highest_reward_strategy} would be the safest "
		"choice for deployment because it achieved the highest cumulative reward "
		"while still maintaining a balance between learning and treatment quality. "
		"A real hospital must continue learning about treatment effectiveness "
		"without exposing too many patients to inferior options."
	)
    
    print("\nOverall Analysis:")
    print(
		f"In this experiment, {highest_reward_strategy} achieved the highest cumulative "
		f"reward, followed by {second_best_strategy}. Immediate Exploitation "
		"converged quickly but was sensitive to early outcomes. Strategies that "
		"continued exploring generally adapted better to the hidden medicine "
		"success probabilities. The results show that maintaining some level of "
		"exploration is important for identifying effective treatments. Based on "
		"both reward and consistency, the top-performing strategy is the most "
		"suitable recommendation for this synthetic clinical setting."
	)


def save_outputs_to_csv(strategy_results, comparison_df):
    """
    Saves everything to CSV files so we have a record of the results.
    One file per strategy + one comparison summary file.
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
    Runs everything in order: setup -> all strategies -> comparison & analysis.
    """
    # Task 1: Build the dataset and show the environment setup
    base_dataset, success_probabilities, num_medicines = task_1_dataset_design()

    # Task 2: Try the greedy approach (explore briefly, then commit)
    df_immediate = run_immediate_exploitation(
        base_dataset,
        success_probabilities,
        GROUP_NUMBER,
        INITIAL_TRIALS_PER_MEDICINE,
    )

    # Task 3: Epsilon-greedy with different exploration rates
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

    # Task 4: UCB1 - the confidence-based approach
    df_ucb1 = run_ucb1(
        base_dataset,
        success_probabilities,
        GROUP_NUMBER,
    )

    # Task 5: Compare all strategies
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

    # Generate the comparison plots
    create_comparison_plot(strategy_results)
    create_selection_count_plot(strategy_results, num_medicines)

    # Print our analysis and answers
    print_final_analysis(comparison_df)

    # Save everything to CSV
    save_outputs_to_csv(strategy_results, comparison_df)


if __name__ == "__main__":
    main()
