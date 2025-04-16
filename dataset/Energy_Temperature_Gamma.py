"""Plot the figure of total energy, potential energy, kinetic energy, temperature, and gamma value based on gamma_data.yaml"""

import matplotlib.pyplot as plt

def plot_energy(mdtrj, save_path_energy=None):
    # Lists to store data from each column
    time = []
    Etot = []
    Epot = []
    Ekin = []
    Temp = []
    # Read data from the file
    with open(mdtrj, 'r') as file:
        next(file)
        for line in file:
            # Split the line into columns based on whitespace
            columns = line.split()

            # Append values to respective columns
            time.append(float(columns[0]))
            Etot.append(float(columns[1]))
            Epot.append(float(columns[2]))
            Ekin.append(float(columns[3]))
            Temp.append(float(columns[4]))

    fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    # Plot the data for each energy component
    axs[0].plot(time, Etot, label='total energy', color='black')
    axs[1].plot(time, Epot, label='potential energy', color='blue')
    axs[2].plot(time, Ekin, label='kinetic energy', color='green')
    axs[3].plot(time, Temp, label='temperature', color='orange')

    # Add labels and title
    axs[3].set_xlabel('time (ps)')
    axs[0].set_ylabel('Total Energy')
    axs[1].set_ylabel('Potential Energy')
    axs[2].set_ylabel('Kinetic Energy')
    axs[3].set_ylabel('Temperature')

    for ax in axs:
        ax.legend()
        ax.grid(True)

    # Show the plot
    plt.tight_layout()

    if save_path_energy is not None:
        plt.savefig(save_path_energy)
    else:
        plt.show()

def plot_gamma(gamma_yaml, gamma_txt, save_path_gamma=None):
    idx = []
    gamma = []

    # Open the file (for example, using the 'with' statement)
    with open(gamma_yaml, 'r') as f1, open(gamma_txt, 'w') as f2:
        for i, line in enumerate(f1, start=1):
            if line.startswith('- index:'):
                index = line.split(":")[-1].strip()  # Assuming the index is the last word in the line
                continue
            elif 'gamma_structure:' in line:
                gamma_structure = line.split(":")[-1].strip()
                f2.write(f"{index}\t{gamma_structure}\n")

                # Append the values to the list for plotting
                idx.append(int(index))
                gamma.append(float(gamma_structure))

    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    # Plot the line plot on the first subplot
    ax1.plot(idx, gamma, marker='o')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Gamma Value')
    ax1.set_title('Line Plot of Gamma Values')
    ax1.grid(True)

    # Plot the histogram on the second subplot
    ax2.hist(gamma, bins=20, color='blue', edgecolor='black')
    ax2.set_xlabel('Gamma Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Gamma Values')
    ax2.grid(True)

#    # Filter data between indices 60000 and 100000
#    filtered_idx = [i for i in idx if 60000 <= i <= 100000]
#    filtered_gamma = [g for i, g in zip(idx, gamma) if 60000 <= i <= 100000]

#    # Plot the histogram using filtered data
#    ax3.hist(filtered_gamma, bins=20, color='blue', edgecolor='black')
#    ax3.set_xlabel('Gamma Value')
#    ax3.set_ylabel('Frequency')
#    ax3.set_title('Histogram of Gamma Values (Index: 60000 to 100000)')
#    ax3.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()

    if save_path_gamma is not None:
        plt.savefig(save_path_gamma)
    else:
        plt.show()

folder = '/Users/Downloads/'
# Specify file paths
mdtrj = folder+ "/md.log"
gamma_yaml = folder+ "/gamma_data.yaml"
gamma_txt = folder+ '/gamma.txt'
# Plot energy
name="Li001_"
save_path_energy = name+"energy.png"
plot_energy(mdtrj, save_path_energy)

# Plot gamma
save_path_gamma = "gamma.png"
plot_gamma(gamma_yaml, gamma_txt, save_path_gamma)
plt.show()
