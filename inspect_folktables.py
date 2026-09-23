"""
Inspect ACS Income dataset structure from folktables.
Run this before the sweep to understand environments, dimensions, class balance.
"""
import numpy as np
from folktables import ACSDataSource, ACSIncome

# training states (9 environments) + 1 test state
TRAIN_STATES = ['CA', 'TX', 'NY', 'FL', 'PA', 'IL', 'OH', 'GA', 'NC']
TEST_STATE   = ['MI']
ALL_STATES   = TRAIN_STATES + TEST_STATE

print("Downloading ACS Income data (2018)...")
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

print("\n--- Training environments ---")
for state in TRAIN_STATES:
    data  = data_source.get_data(states=[state], download=True)
    X, y, _ = ACSIncome.df_to_numpy(data)
    print(f"  {state}: n={len(X):>6}  n_features={X.shape[1]}"
          f"  class_balance={y.mean():.3f}")

print("\n--- Test environment ---")
data  = data_source.get_data(states=TEST_STATE, download=True)
X, y, _ = ACSIncome.df_to_numpy(data)
print(f"  {TEST_STATE[0]}: n={len(X):>6}  n_features={X.shape[1]}"
      f"  class_balance={y.mean():.3f}")

print("\n--- Feature names ---")
print(ACSIncome.features)

print("\n--- Summary ---")
print(f"  n_train_envs : {len(TRAIN_STATES)}")
print(f"  n_test_envs  : {len(TEST_STATE)}")
print(f"  n_features   : {X.shape[1]}")
print(f"  n_classes    : 2 (income >= 50k)")
print(f"  shift type   : geographic covariate shift")
print(f"  expected Cross-R: positive (misspecified)")