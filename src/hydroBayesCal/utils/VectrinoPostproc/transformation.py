"""Retrieve transformation matrix to transform beam signals into velocities with Cartesian coordinates"""
import numpy as np
import pandas as pd


def get_transformation_matrix(ntk_file_name, scaling_factor=4096):
    """ Read the transformation matrix as numpy array from the

    :param str ntk_file_name: Path and name of a Vectrino II ntk file. For example 'data/VectrinoProfiler.00001' when
                            the header file is 'VectrinoProfiler.00001.ntk.hdr' (see 'sample-data/' directory).
    :param int scaling_factor: This number can be found in the hdr file on line 186 (or so) and is typically device
                                specific.
    :return np.array M: The transformation matrix.
    """
    # initialize hBeamToXYZ
    hBeamToXYZ = []
    # retrieve hBeamToXYZ values from header file
    with open(ntk_file_name.strip('.ntk.hdr') + '.ntk.hdr', 'r') as file:
        for line in file:
            if 'Probe_hBeamToXYZ' in line:
                # extract probe transformation
                line_str_list = line.split(':')
                if len(line_str_list) > 1:
                    # this is required to avoid that the description lower in the header file is mistaken as transformation
                    Probe_hBeamToXYZ = line_str_list[-1].strip().strip('[]').split()
                    hBeamToXYZ = [float(e) for e in Probe_hBeamToXYZ]

    if len(hBeamToXYZ) > 0:
        # convert hBeamToXYZ list into a 4x4 np.array (matrix) by dividing each element by the scaling factor
        return np.array(hBeamToXYZ).reshape(4, 4) / scaling_factor
    else:
        # error!
        print('ERROR: Could not read transformation matrix from the header file.')
        return -1


def apply_transformation(
        df,
        transformation_matrix,
        relevant_point_ids=(0, 1, 2),
):
    """Apply beam-to-XYZ transformation, or pass through already-XYZ Vectrino data.

    Supports two input formats:

    1. Beam format:
       Velocity Beam 1 (m/s), Velocity Beam 2 (m/s),
       Velocity Beam 3 (m/s), Velocity Beam 4 (m/s)

       In this case, the transformation matrix is applied.

    2. XYZ format:
       Velocity X (m/s), Velocity Y (m/s),
       Velocity Z1 (m/s), Velocity Z2 (m/s)

       In this case, no transformation is applied. The function only averages
       the selected profiler points and creates:
       u (m/s), v (m/s), w1 (m/s), w2 (m/s)

    The function keeps the same interface, so the main script does not need
    to be changed.
    """

    if len(relevant_point_ids) == 0:
        raise ValueError('ERROR: Invalid argument for list relevant_point_ids provided.')

    if df.empty:
        raise ValueError(
            'ERROR: Input DataFrame is empty. '
            'The Vectrino ASCII reader probably did not parse this file correctly.'
        )

    beam_cols = {
        'b1': 'Velocity Beam 1 (m/s)',
        'b2': 'Velocity Beam 2 (m/s)',
        'b3': 'Velocity Beam 3 (m/s)',
        'b4': 'Velocity Beam 4 (m/s)',
    }

    xyz_cols = {
        'x': 'Velocity X (m/s)',
        'y': 'Velocity Y (m/s)',
        'z1': 'Velocity Z1 (m/s)',
        'z2': 'Velocity Z2 (m/s)',
    }

    has_beam_data = all(col in df.columns for col in beam_cols.values())
    has_xyz_data = all(col in df.columns for col in xyz_cols.values())

    if not has_beam_data and not has_xyz_data:
        raise ValueError(
            'ERROR: Could not detect velocity format.\n'
            'Expected either beam-coordinate columns:\n'
            f'{list(beam_cols.values())}\n'
            'or XYZ-coordinate columns:\n'
            f'{list(xyz_cols.values())}\n'
            'Available columns are:\n'
            f'{list(df.columns)}'
        )

    if has_beam_data:
        velocity_cols = [
            beam_cols['b1'],
            beam_cols['b2'],
            beam_cols['b3'],
            beam_cols['b4'],
        ]

        points_per_measurement = len(df.iloc[0][beam_cols['b1']])

        print('   - found ' + str(points_per_measurement) + ' points per measurement')
        print('   - beam-coordinate data detected')
        print('   - applying beam-to-XYZ transformation...')

    else:
        velocity_cols = [
            xyz_cols['x'],
            xyz_cols['y'],
            xyz_cols['z1'],
            xyz_cols['z2'],
        ]

        points_per_measurement = len(df.iloc[0][xyz_cols['x']])

        print('   - found ' + str(points_per_measurement) + ' points per measurement')
        print('   - XYZ-coordinate data detected')
        print('   - skipping beam-to-XYZ transformation...')

    invalid_ids = [pid for pid in relevant_point_ids if pid >= points_per_measurement or pid < 0]

    if invalid_ids:
        raise ValueError(
            'ERROR: relevant_point_ids contains invalid point IDs.\n'
            f'Available point IDs: 0 to {points_per_measurement - 1}\n'
            f'Requested point IDs: {relevant_point_ids}\n'
            f'Invalid point IDs: {invalid_ids}'
        )

    u = []
    v = []
    w1 = []
    w2 = []

    for row_id in range(len(df)):

        values_1 = df.loc[row_id, velocity_cols[0]]
        values_2 = df.loc[row_id, velocity_cols[1]]
        values_3 = df.loc[row_id, velocity_cols[2]]
        values_4 = df.loc[row_id, velocity_cols[3]]

        u_values = []
        v_values = []
        w1_values = []
        w2_values = []

        for point_id in range(points_per_measurement):

            raw_velocity_vector = np.array([
                values_1[point_id],
                values_2[point_id],
                values_3[point_id],
                values_4[point_id],
            ])

            if has_beam_data:
                cartesian_velocities = np.dot(
                    transformation_matrix[:4, :],
                    raw_velocity_vector
                )
            else:
                cartesian_velocities = raw_velocity_vector

            u_values.append(cartesian_velocities[0])
            v_values.append(cartesian_velocities[1])
            w1_values.append(cartesian_velocities[2])
            w2_values.append(cartesian_velocities[3])

        u.append(np.mean([u_values[point_id] for point_id in relevant_point_ids]))
        v.append(np.mean([v_values[point_id] for point_id in relevant_point_ids]))
        w1.append(np.mean([w1_values[point_id] for point_id in relevant_point_ids]))
        w2.append(np.mean([w2_values[point_id] for point_id in relevant_point_ids]))

    df['u (m/s)'] = u
    df['v (m/s)'] = v
    df['w1 (m/s)'] = w1
    df['w2 (m/s)'] = w2

    return df
