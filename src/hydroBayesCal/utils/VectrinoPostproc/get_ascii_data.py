"""
Read flow velocity data from Nortek Vectrino II / Vectrino Profiler ASCII output (.ntk.dat) files.

Supports:
    - Beam-coordinate files:
        Profiles_Velocity_Beam_1 ... Profiles_Velocity_Beam_4

    - XYZ-coordinate files:
        Profiles_Velocity_X
        Profiles_Velocity_Y
        Profiles_Velocity_Z1
        Profiles_Velocity_Z2
"""

import numpy as np
import pandas as pd


def read_ascii_file(file_name):
    """Read the contents of a Vectrino ASCII .ntk.dat file.

    :param str file_name: Path and name without .ntk.dat extension.
                          Example: 'sample-data/VectrinoProfiler.00001'
    :return pd.DataFrame: DataFrame containing either beam velocities or XYZ velocities,
                          plus SNR values if available.
    """

    # initialize common storage
    time_list = []

    # beam velocity storage
    velocity_beam_1 = []
    velocity_beam_2 = []
    velocity_beam_3 = []
    velocity_beam_4 = []

    # XYZ velocity storage
    velocity_x = []
    velocity_y = []
    velocity_z1 = []
    velocity_z2 = []

    # SNR storage
    snr_beam_1 = []
    snr_beam_2 = []
    snr_beam_3 = []
    snr_beam_4 = []

    # detect which velocity format was parsed
    parsed_beam_data = False
    parsed_xyz_data = False

    dat_file = file_name.strip('.ntk.dat') + '.ntk.dat'
    print('   - reading ' + dat_file)

    with open(dat_file, 'r', encoding='latin1', errors='ignore') as file:
        current_time = None
        current_velocities_beam = {}
        current_velocities_xyz = {}
        current_snr = {}

        for line in file:

            # extract time
            if 'Profiles_HostTime_start (s)' in line:
                current_time = float(line.split(':')[-1].strip())

            # ----------------------------------------------------------
            # Beam-coordinate velocity lines
            # ----------------------------------------------------------
            elif 'Profiles_Velocity_Beam_' in line:
                beam_number = int(
                    line.split('Profiles_Velocity_Beam_')[1].split()[0]
                )

                velocities = line.split(':')[-1].strip().strip('[]').split()
                current_velocities_beam[beam_number] = [float(v) for v in velocities]

            # ----------------------------------------------------------
            # XYZ-coordinate velocity lines
            # ----------------------------------------------------------
            elif 'Profiles_Velocity_X' in line:
                velocities = line.split(':')[-1].strip().strip('[]').split()
                current_velocities_xyz['x'] = [float(v) for v in velocities]

            elif 'Profiles_Velocity_Y' in line:
                velocities = line.split(':')[-1].strip().strip('[]').split()
                current_velocities_xyz['y'] = [float(v) for v in velocities]

            elif 'Profiles_Velocity_Z1' in line:
                velocities = line.split(':')[-1].strip().strip('[]').split()
                current_velocities_xyz['z1'] = [float(v) for v in velocities]

            elif 'Profiles_Velocity_Z2' in line:
                velocities = line.split(':')[-1].strip().strip('[]').split()
                current_velocities_xyz['z2'] = [float(v) for v in velocities]

            # ----------------------------------------------------------
            # SNR lines
            # ----------------------------------------------------------
            elif 'Profiles_SNR_Beam_' in line:
                beam_number = int(
                    line.split('Profiles_SNR_Beam_')[1].split()[0]
                )

                snr = line.split(':')[-1].strip().strip('[]').split()
                current_snr[beam_number] = np.nanmean([float(sig) for sig in snr])

            # ----------------------------------------------------------
            # Append complete Beam record
            # ----------------------------------------------------------
            if len(current_velocities_beam) == 4:
                parsed_beam_data = True

                time_list.append(current_time)

                velocity_beam_1.append(current_velocities_beam[1])
                velocity_beam_2.append(current_velocities_beam[2])
                velocity_beam_3.append(current_velocities_beam[3])
                velocity_beam_4.append(current_velocities_beam[4])

                if len(current_snr) == 4:
                    snr_beam_1.append(current_snr[1])
                    snr_beam_2.append(current_snr[2])
                    snr_beam_3.append(current_snr[3])
                    snr_beam_4.append(current_snr[4])
                else:
                    snr_beam_1.append(np.nan)
                    snr_beam_2.append(np.nan)
                    snr_beam_3.append(np.nan)
                    snr_beam_4.append(np.nan)

                current_velocities_beam = {}
                current_snr = {}

            # ----------------------------------------------------------
            # Append complete XYZ record
            # ----------------------------------------------------------
            if len(current_velocities_xyz) == 4:
                parsed_xyz_data = True

                time_list.append(current_time)

                velocity_x.append(current_velocities_xyz['x'])
                velocity_y.append(current_velocities_xyz['y'])
                velocity_z1.append(current_velocities_xyz['z1'])
                velocity_z2.append(current_velocities_xyz['z2'])

                if len(current_snr) == 4:
                    snr_beam_1.append(current_snr[1])
                    snr_beam_2.append(current_snr[2])
                    snr_beam_3.append(current_snr[3])
                    snr_beam_4.append(current_snr[4])
                else:
                    snr_beam_1.append(np.nan)
                    snr_beam_2.append(np.nan)
                    snr_beam_3.append(np.nan)
                    snr_beam_4.append(np.nan)

                current_velocities_xyz = {}
                current_snr = {}

    # ------------------------------------------------------------------
    # Return Beam-format DataFrame
    # ------------------------------------------------------------------
    if parsed_beam_data and not parsed_xyz_data:
        return pd.DataFrame({
            'Time (s)': time_list,
            'Velocity Beam 1 (m/s)': velocity_beam_1,
            'Velocity Beam 2 (m/s)': velocity_beam_2,
            'Velocity Beam 3 (m/s)': velocity_beam_3,
            'Velocity Beam 4 (m/s)': velocity_beam_4,
            'SNR Beam 1 (dB)': snr_beam_1,
            'SNR Beam 2 (dB)': snr_beam_2,
            'SNR Beam 3 (dB)': snr_beam_3,
            'SNR Beam 4 (dB)': snr_beam_4,
        })

    # ------------------------------------------------------------------
    # Return XYZ-format DataFrame
    # ------------------------------------------------------------------
    elif parsed_xyz_data and not parsed_beam_data:
        return pd.DataFrame({
            'Time (s)': time_list,
            'Velocity X (m/s)': velocity_x,
            'Velocity Y (m/s)': velocity_y,
            'Velocity Z1 (m/s)': velocity_z1,
            'Velocity Z2 (m/s)': velocity_z2,
            'SNR Beam 1 (dB)': snr_beam_1,
            'SNR Beam 2 (dB)': snr_beam_2,
            'SNR Beam 3 (dB)': snr_beam_3,
            'SNR Beam 4 (dB)': snr_beam_4,
        })

    # ------------------------------------------------------------------
    # Mixed or unreadable case
    # ------------------------------------------------------------------
    elif parsed_beam_data and parsed_xyz_data:
        raise ValueError(
            'ERROR: File contains both beam-coordinate and XYZ-coordinate velocity records. '
            'This reader expects only one velocity format per file.'
        )

    else:
        raise ValueError(
            'ERROR: No valid velocity records found in file: ' + dat_file
        )
