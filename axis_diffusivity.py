from ase import Atoms
from ase.io import read
from ase.md.analysis import DiffusionCoefficient
from ase.units import fs as fs_conversion
import numpy as np
import pandas as pd
import yaml


class DiffusionAnalyzer:
    """
    Class to compute the diffusion coefficient using mean square displacement (MSD)
    for a given trajectory of configurations.0
    """

    def __init__(self, traj, dt, trajfreq, ignore_n_images, number_of_segments, step_frame=1):
        """
        Initialize the DiffusionAnalyzer class with the necessary data and parameters.

        Args:
            traj: Trajectory (ASE Atoms object or list of Atoms objects)
            dt: MD simulation timestep in  ps
            trajfreq: Trajectory recorded frequency (in frames)
            step_frame: Frame step to use when loading (default is 1)
            ignore_n_images: Number of initial images to ignore for diffusion calculation
            number_of_segments: Number of segments for MSD computation
        """
        if isinstance(traj, list) and isinstance(traj[0], Atoms):
            self.images = traj  # list of ASE Atoms objects (trajectory)
        elif isinstance(traj, Atoms):
            self.images = [traj]  # single ASE Atoms object
        else:
            raise ValueError("Invalid trajectory input")

        self.dt = dt
        self.trajfreq = trajfreq
        self.step_frame = step_frame
        self.ignore_n_images = ignore_n_images
        self.number_of_segments = number_of_segments
        self.timestep = float(self.dt * 1000 * self.trajfreq * self.step_frame * fs_conversion)
        self.dc = DiffusionCoefficient(self.images, self.timestep)

        # Initialize placeholders
        self.msd_data = None
        self.diffusion_coefficients = None


    def compute_diffusion_coefficient(self, save=True, json_filename="diffusion_results.json"):
        """
        Calculate the diffusion coefficients in x, y, z, xy (ab-plane), and xyz (total).
        """
        self.dc.calculate(self.ignore_n_images, self.number_of_segments)
        species = self.dc.types_of_atoms
        slopes_xyz_segments = self.dc.slopes

        for i, element in enumerate(slopes_xyz_segments):
            # Unit conversion to cm^2/s
            dx = np.mean(element[:, 0]) * fs_conversion * 1e-1  # cm^2/s
            dy = np.mean(element[:, 1]) * fs_conversion * 1e-1  # cm^2/s
            dz = np.mean(element[:, 2]) * fs_conversion * 1e-1  # cm^2/s

            dxyz = np.mean([dx, dy, dz])  # This should be the same as 'dc.print_data'
            dxy = np.mean([dx, dy])  # Diffusion in the ab-plane

            # Print the diffusion coefficients
            print(f"{species[i]}")
            print(f"D_x  : {dx: .2e} cm^2/s")
            print(f"D_y  : {dy: .2e} cm^2/s")
            print(f"D_z  : {dz: .2e} cm^2/s")
            print(f"D_xy : {dxy: .2e} cm^2/s")
            print(f"D_xyz: {dxyz: .2e} cm^2/s\n")


    def compute_msd_data(self):
        """
        Calculate the Mean Square Displacement (MSD) for the given trajectory and store the data.
        """
        species = self.dc.types_of_atoms
        len_segments = self.dc.len_segments

        time = []
        msd = {species[i]: {"x": [], "y": [], "z": []} for i in range(len(species))}

        for i in range(len_segments):
            t = i * self.timestep / fs_conversion / 1000.0  # convert fs to ps
            time.append(t)

            msd_avg_segment = [0.0] * (len(species) * 3)

            for j in range(len(species)):
                for k in range(3):
                    msd_segment = []  # collect all the data for this component

                    for segment_no in range(self.number_of_segments):
                        msd_segment.append(self.dc.xyz_segment_ensemble_average[segment_no][j][k][i])

                    msd_avg_segment[j * 3 + k] = np.mean(msd_segment)

                # Store the MSD data for each species in x, y, z directions
                msd[species[j]]["x"].append(msd_avg_segment[j * 3 + 0])  # x component
                msd[species[j]]["y"].append(msd_avg_segment[j * 3 + 1])  # y component
                msd[species[j]]["z"].append(msd_avg_segment[j * 3 + 2])  # z component

        self.msd_data = (time, msd)
        return self.msd_data


    def save_msd_data(self, filename="msd_data.txt"):
        """
        Save the MSD data as a plain text file.
        """
        time, msd = self.msd_data

        with open(filename, 'w') as f:
            # Write header
            f.write("Time(ps) " + " ".join([f"{s}_x {s}_y {s}_z" for s in msd]) + "\n")

            # Write MSD data for each time step
            for i, t in enumerate(time):
                msd_line = f"{t:.4f} "  # Time in ps
                for species_name in msd:
                    msd_line += f"{msd[species_name]['x'][i]:.6f} "  # x component
                    msd_line += f"{msd[species_name]['y'][i]:.6f} "  # y component
                    msd_line += f"{msd[species_name]['z'][i]:.6f} "  # z component
                f.write(msd_line.strip() + "\n")

            print(f"MSD data saved as {filename}")

    def plot_msd(self, filename="msd_plot.pdf", show=False):
        """
        Plot the MSD vs time for each species.

        Args:
            filename: Name of the file to save the plot.
            save: If True, the plot will be saved to the file.
            show: If True, the plot will be displayed.
        """

        time, msd = self.msd_data
        species = self.dc.types_of_atoms

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(species), figsize=(18, 6))
        if len(species) == 1:
            axes = [axes]

        for i, species_name in enumerate(species):
            ax = axes[i]
            ax.plot(time, msd[species_name]["x"], label=f"{species_name} (x)", linestyle='-', marker='o')
            ax.plot(time, msd[species_name]["y"], label=f"{species_name} (y)", linestyle='-', marker='s')
            ax.plot(time, msd[species_name]["z"], label=f"{species_name} (z)", linestyle='-', marker='^')

            ax.set_xlabel("Time (ps)", fontsize=20)
            ax.set_ylabel("MSD (Å²)", fontsize=20)
            ax.set_title(f"MSD for {species_name}", fontsize=16)
            ax.legend(fontsize=16)

        plt.tight_layout()
        plt.savefig(filename)

        if show:
            plt.show()


def load_trajectory(file_path, trajlen, dt, trajfreq, step_frame=1):
    """
    Load a trajectory from a file.

    Args:
        file_path: Path to the trajectory file.
        trajlen: Length of the trajectory in ps.
        dt: MD simulation timestep in ps.
        trajfreq: Trajectory recorded frequency (in frames).
        step_frame: Frame step to use when loading (default is 1).

    Returns:
        List of ASE Atoms objects representing the trajectory.
    """
    start_frame = 3000
    end_frame = int(trajlen / dt / trajfreq)
    traj = read(file_path, index=f"{start_frame}:{end_frame}:{step_frame}")
    return traj


def main():

    traj_file_path = "md.traj"
    trajlen = 1000
    dt = 0.001
    trajfreq = 100
    step_frame = 1

    # Load the trajectory
    traj = load_trajectory(traj_file_path, trajlen, dt, trajfreq, step_frame)

    # Initialize the DiffusionAnalyzer class
    ignore_n_images = 0  # Number of frames to ignore
    number_of_segments = 1
    analyzer = DiffusionAnalyzer(traj, dt, trajfreq, ignore_n_images, number_of_segments, step_frame)

    # Compute diffusion coefficient and MSD
    analyzer.compute_diffusion_coefficient()
    analyzer.compute_msd_data()
    analyzer.save_msd_data()
    analyzer.plot_msd(show=True)


if __name__ == "__main__":
    main()
