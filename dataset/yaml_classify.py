import yaml

def split_bulk_slab(yaml_path, bulk_file, slab_file, z_threshold=10.0):
    """
    Split structures in a YAML file into bulk and slab categories based on vacuum thickness.
    :param yaml_path: Path to the original YAML file
    :param bulk_file: Output file for bulk structures
    :param slab_file: Output file for slab structures
    :param z_threshold: Minimum vacuum thickness (in Å) to classify as slab
    """
    # Load the original YAML file
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Check the format of the YAML file
    if isinstance(data, dict) and 'data_points' in data:
        entries = data['data_points']
        wrap_key = 'data_points'
    elif isinstance(data, list):
        entries = data
        wrap_key = None
    else:
        raise ValueError("Unexpected YAML format. Please check the file.")

    bulk_entries = []
    slab_entries = []

    # Classify each entry
    for entry in entries:
        structure = entry.get('structure', {})
        lattice = structure.get('lattice', {})
        sites = structure.get('sites', [])
        c_length = lattice.get('c', 0)

        if not sites or not c_length:
            continue
        try:
            z_coords = [site['abc'][2] for site in sites]
        except Exception:
            continue

        cartesian_z = [z * c_length for z in z_coords]
        z_span = max(cartesian_z) - min(cartesian_z)
        vacuum_estimate = c_length - z_span

        if vacuum_estimate > z_threshold:
            slab_entries.append(entry)
        else:
            bulk_entries.append(entry)

    # Save results to YAML files, preserving original format
    with open(slab_file, "w") as f:
        yaml.dump({wrap_key: slab_entries} if wrap_key else slab_entries, f, sort_keys=False)

    with open(bulk_file, "w") as f:
        yaml.dump({wrap_key: bulk_entries} if wrap_key else bulk_entries, f, sort_keys=False)

    print(f"Successfully saved {len(slab_entries)} slab structures and {len(bulk_entries)} bulk structures.")


# Example usage
split_bulk_slab(
    yaml_path="dataset_dftmd_test.yaml",
    bulk_file="dataset_dftmd_test_bulk.yaml",
    slab_file="dataset_dftmd_test_slab.yaml",
    z_threshold=10.0
)
