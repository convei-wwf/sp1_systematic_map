import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Check if file is passed as a command line argument
if len(sys.argv) < 2:
    print("Please provide the CSV file name.")
    sys.exit(1)

# Load the CSV file
csv_file = sys.argv[1]
data = pd.read_csv(csv_file + '.csv')

# Create the plot
plt.figure()
plt.plot(data['epoch'], data['accuracy'], label='Accuracy', color='blue')
plt.plot(data['epoch'], data['loss'], label='Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Accuracy and Loss over Epochs')
plt.legend()

# Save the plot as a PNG file
output_file = os.path.splitext(csv_file)[0] + ".png"
plt.savefig(output_file)

print(f"Graph saved as {output_file}")
