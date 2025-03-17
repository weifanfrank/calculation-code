from ase.io import read
from ase.md.analysis import DiffusionCoefficient
from ase.units import fs as fs_conversion
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


# MD simulation details
trajfreq = 10000  # trajectory recorded frequency
trajlen = 1000  # ps, trajectory length
dt = 0.001  # ps, MD simulation timestep


def load_trajectory(file_path, start, end, step):
    """Load the trajectory."""
    traj = read(file_path, index=f"{start}:{end}:{step}")
    print(f"Number of frames in the trajectory: {len(traj)}")
    return traj


def get_unique_species(traj):
    """Extract unique species from the trajectory."""
    return sorted(set(traj[0].symbols))


def initialize_diffusion_coefficient(traj, timestep, atom_indices=None, molecule=False):
    """Initialize the DiffusionCoefficient object."""
    return DiffusionCoefficient(traj, timestep, atom_indices, molecule)


def calculate_diffusion_coefficient(diff_coeff, ignore_n_images, number_of_segments):
    """Calculate the diffusion coefficient."""
    diff_coeff.calculate(ignore_n_images, number_of_segments)
    slopes, std = diff_coeff.get_diffusion_coefficients()
    return slopes, std


def print_diffusion_coefficients(slopes, std, atom_types):
    """Print the diffusion coefficients and standard deviations."""
    print("Diffusion coefficients:")
    for i, (slope, std_dev) in enumerate(zip(slopes, std)):
        print(
            f"Species {atom_types[i]}: {slope * fs_conversion * 1000:.5f} +/- {std_dev * fs_conversion * 1000:.5f} Å^2/ps"
        )


def plot_msd_data(diff_coeff, plot_path, unique_species):
    """Plot MSD data and save the plot."""
    # Extract necessary data from diff_coeff object
    time = diff_coeff.timesteps / fs_conversion / 1000  # Convert to ps
    species_data = diff_coeff.xyz_segment_ensemble_average

    # Plotting setup
    plt.figure(figsize=(10, 6))

    # Plot each species MSD data and linear fit
    for sym_index in range(diff_coeff.no_of_types_of_atoms):
        msd_values = np.mean(
            species_data[:, sym_index, :, :], axis=(0, 1)
        )  # Average over segments, xyz

        # Ensure time and msd_values have the same length
        min_length = min(len(time), len(msd_values))
        time = time[:min_length]
        msd_values = msd_values[:min_length]

        # Remove any NaN values
        valid_indices = ~np.isnan(msd_values)
        time_valid = time[valid_indices]
        msd_valid = msd_values[valid_indices]

        if len(time_valid) > 1:  # Ensure we have at least two points for regression
            # Perform linear regression (fitting a line)
            slope, intercept, r_value, p_value, std_err = linregress(
                time_valid, msd_valid
            )
            fit_line = slope * time + intercept

            # Plot MSD data
            plt.plot(
                time,
                msd_values,
                marker="^",
                label=f"{ unique_species[sym_index]} (data)",
            )
            plt.plot(time, fit_line, "--", label=f"{ unique_species[sym_index]} (fit)")
        else:
            print(f"Warning: Not enough valid data points for Species {sym_index + 1}")

    # Finalize plot details
    plt.xlabel("Time (ps)", fontsize=14)
    plt.ylabel("MSD (Å^2)", fontsize=14)
    plt.title("Mean Square Displacement (MSD) vs Time", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Save the plot
    plt.savefig(plot_path)
    plt.close()  # Close the figure to release memory
    print(f"Plot saved as {plot_path}")


def save_msd_data(diff_coeff, msd_file_path, unique_species):
    """Save MSD data to a file."""
    with open(msd_file_path, "w") as file:
        # Create the header line
        header = "Time(ps)"
        for species in unique_species:
            header += f" {species}_x {species}_y {species}_z"
        file.write(header + "\n")

        for image_no in range(len(diff_coeff.timesteps)):
            t = diff_coeff.timesteps[image_no] / fs_conversion / 1000.0  # convert fs to ps

            msd_values = [0.0] * (diff_coeff.no_of_types_of_atoms * 3)
            count_valid_segments = 0

            for sym_index in range(diff_coeff.no_of_types_of_atoms):
                for xyz in range(3):
                    msd_segment_values = [
                        diff_coeff.xyz_segment_ensemble_average[segment_no][sym_index][
                            xyz
                        ][image_no]
                        for segment_no in range(diff_coeff.no_of_segments)
                        if image_no
                        < len(
                            diff_coeff.xyz_segment_ensemble_average[segment_no][
                                sym_index
                            ][xyz]
                        )
                    ]

                    if msd_segment_values:
                        msd_values[sym_index * 3 + xyz] = np.mean(msd_segment_values)
                        count_valid_segments += 1
                    else:
                        msd_values[sym_index * 3 + xyz] = np.nan

            if count_valid_segments > 0:
                msd_line = f"{t:.4f} " + " ".join(
                    [f"{msd:.6f}" if not np.isnan(msd) else "nan" for msd in msd_values]
                )
                file.write(msd_line + "\n")

    print(f"MSD data saved as {msd_file_path}")


def main():
    traj_file_path = "md.traj"
    plot_path = "diffusion_coefficient.png"
    msd_file_path = "msd_data.txt"
    start_frame = 0
    end_frame = int(trajlen / dt / trajfreq)
    step_frame = 1
    timestep_fs = float(trajfreq * step_frame * fs_conversion)
    ignore_n_images = 0
    number_of_segments = 1

    # Load the trajectory
    traj = load_trajectory(traj_file_path, start_frame, end_frame, step_frame)

    # Get unique species from the trajectory
    unique_species = get_unique_species(traj)
    print(f"Unique species in the trajectory: {unique_species}")

    # Initialize the DiffusionCoefficient object
    diff_coeff = initialize_diffusion_coefficient(traj, timestep_fs)

    # Calculate the diffusion coefficient
    slopes, std = calculate_diffusion_coefficient(
        diff_coeff, ignore_n_images, number_of_segments
    )

    # Print the diffusion coefficients
    print_diffusion_coefficients(slopes, std, unique_species)

    # Plot and save the diffusion coefficient
    plot_msd_data(diff_coeff, plot_path, unique_species)

    # Save the MSD data
    save_msd_data(diff_coeff, msd_file_path, unique_species)


if __name__ == "__main__":
    main()
