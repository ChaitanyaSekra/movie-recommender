import pandas as pd
import re

# Load your CSV
csv_file = "updated_file.csv"  # Replace with your actual filename
df = pd.read_csv(csv_file)
'''
# Define the conversion function
def convert_duration_to_minutes(duration_str):
    if pd.isna(duration_str):
        return None
    hours = re.search(r'(\d+)\s*hour', duration_str)
    minutes = re.search(r'(\d+)\s*minute', duration_str)
    total = 0
    if hours:
        total += int(hours.group(1)) * 60
    if minutes:
        total += int(minutes.group(1))
    return total

# Apply the function to the Duration column
df["DurationInMinutes"] = df["Duration"].apply(convert_duration_to_minutes)

# Save to a new CSV (or overwrite)
df.to_csv("updated_file.csv", index=False)
# Or overwrite the original file:
# df.to_csv(csv_file, index=False)

print("Duration converted and CSV updated.") '''


df["Rating"] = df["Rating"].astype(float)
df["Year"] = df["Year"].astype(int)
df["Duration"] = df["Duration"].astype(int)

df.to_csv("converted_file.csv", index=False)

type(df["Duration"])