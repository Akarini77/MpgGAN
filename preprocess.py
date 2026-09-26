import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# File paths
# ============================================================

input_file = "data/example_input.csv"
output_file = "data/preprocessed_data.csv"


# ============================================================
# Read data
# ============================================================

df = pd.read_csv(
    input_file,
    header=0,
    names=[
        "dip_direction",
        "dip_angle",
        "spacing",
        "trace_length"
    ]
)


# ============================================================
# Min-Max normalization
# ============================================================

scaler = MinMaxScaler()

columns = [
    "dip_direction",
    "dip_angle",
    "spacing",
    "trace_length"
]

df[columns] = scaler.fit_transform(df[columns])

# Keep six decimal places
df[columns] = df[columns].round(6)


# ============================================================
# Save normalized data
# ============================================================

df.to_csv(
    output_file,
    index=False,
    header=False
)

print(f"Input file: {input_file}")
print(f"Output file: {output_file}")