import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_estimations(file_path):
    estimations = []
    with open(file_path, 'r') as file:
        content = file.read()
        pattern = r"Estimation in repetition \d+:\s*\[([\d\.,\s]+)\]"
        matches = re.findall(pattern, content)
        for match in matches:
            estimation = [float(num) for num in match.split(',')]
            estimations.append(estimation)
    return estimations


file_path = './simres/nelder-mead_server.txt'  # Replace with your actual file path
estimations = extract_estimations(file_path)

# Print the extracted estimations
for i, estimation in enumerate(estimations, 1):
    print(f"Estimation in repetition {i}: {estimation}")

# Create a DataFrame for easier plotting
df = pd.DataFrame(estimations).T
df.columns = [f'Rep {i+1}' for i in range(len(estimations))]

""" # Create boxplot
plt.figure(figsize=(10, 6))
df.boxplot()
plt.title('Boxplot of Estimations')
plt.ylabel('Estimation Values')
plt.xlabel('Repetitions')
plt.savefig('estimation_boxplot.png')
plt.show() """