import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

folder_path = '../Data/Raw'
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df_list.append(df)

combined_df = pd.concat(df_list)

plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
plt.plot(combined_df['Thermometer'], label='')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Resampled Data Over Time')
plt.legend()
plt.grid(True)

plt.show()