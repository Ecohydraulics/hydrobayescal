## Vectrino background

The Vectrino II is an acoustic Doppler velocimeter (ADV) for measuring three-dimensional water velocity profiles. It uses multiple acoustic beams to measure the Doppler shift, from which it calculates velocity components. The Vectrino II typically has four beams, oriented in different directions. Each beam measures flow velocity along the line of sight of each beam, referred to as the beam velocities. In this context, **Beam 1 to Beam 4** are the raw velocities measured along the direction of each beam, which are not aligned with the standard Cartesian coordinate system `(x, y, z)` and require conversion.

### Conversion to Cartesian coordinates

The Vectrino Profiler output can be stored in different coordinate systems depending on the export settings. In this workflow, two cases are handled:

1. **Beam-coordinate output**
2. **XYZ-coordinate output**

---

#### Beam-coordinate output

For beam-coordinate files, the measured beam velocities \((B_1, B_2, B_3, B_4)\) are converted into Cartesian velocities using the transformation matrix stored in the `.ntk.hdr` file.

The transformation is:

\[
\begin{pmatrix}
u \\
v \\
w_1 \\
w_2
\end{pmatrix}
=
\mathbf{M}
\begin{pmatrix}
B_1 \\
B_2 \\
B_3 \\
B_4
\end{pmatrix}
\]

where \(u\), \(v\), \(w_1\), and \(w_2\) are the Cartesian velocity components. The two vertical velocity estimates, \(w_1\) and \(w_2\), are independent redundant estimates of the vertical velocity.

The transformation matrix \(\mathbf{M}\) is obtained from the header variable:

#### XYZ-coordinate output

For XYZ-coordinate files, the transformation step is not required, because the velocities are already provided in Cartesian coordinates.

The data directly contains:

Profiles_Velocity_X
Profiles_Velocity_Y
Profiles_Velocity_Z1
Profiles_Velocity_Z2

The processing simply maps these values into the Cartesian velocity components.
### Description of Vectrino Profiler header information

The `.ntk.hdr` file contains the configuration and metadata associated with the Vectrino Profiler measurements. The most relevant parameters for data interpretation are summarized below.

---

#### Measurement configuration (example)

- **Instrument**: Vectrino Profiler (Acoustic Doppler Velocimeter) 
- **Sampling frequency**: 50 Hz 
- **Measurement duration**: ~180 s 
- **Number of samples (nSets)**: 10126 
- **Number of cells (nCells)**: 1 

The measurement was performed at a single vertical location, with:

- **Cell center position**: 42 mm from the probe 
- **Cell size**: ~3.05 mm 

---

#### Coordinate system


coordSystem : 1

### Description of the Vectrino Profiler `.ntk.dat` file

The `.ntk.dat` file contains the time-resolved measurement records exported by the Vectrino Profiler. In this dataset, the file is written in an ASCII block format, where each measurement block contains velocity data, signal quality indicators, and auxiliary acquisition information.

The file contains three main types of blocks:

1. **VelocityHeader blocks**
2. **Profiles blocks**
3. **BottomCheck blocks**

---
#### VelocityHeader block

The `VelocityHeader` block contains general information associated with the beginning of the velocity record. It includes initial signal quality, timing, temperature, speed of sound, and velocity range information.

Example variables are:

```text
VelocityHeader_HostTime_start
VelocityHeader_Correlation_Beam_1
VelocityHeader_Amplitude_Beam_1
VelocityHeader_TimeStamp
VelocityHeader_Temperature
VelocityHeader_Speed_Of_Sound
VelocityHeader_Horizontal_Velocity_Range
VelocityHeader_Vertical_Velocity_Range
```

The .ntk.dat file contains instantaneous velocity measurements and quality-control variables in block format. For this dataset, the velocity data are already stored in XYZ coordinates and consist of one measurement cell per time step. The relevant variables for velocity and turbulence analysis are:

```text
Profiles_Velocity_X
Profiles_Velocity_Y
Profiles_Velocity_Z1
Profiles_Velocity_Z2
Profiles_Correlation_Beam_1 ... Beam_4
Profiles_SNR_Beam_1 ... Beam_4
Profiles_TimeStamp
```

More information can be found in the [Vectrino II manual](https://www.nortekgroup.com/assets/software/N3015-030-Comprehensive-Manual-Velocimeters_1118.pdf).

### Quality considerations

Ensure your instrument is **correctly calibrated**, as incorrect calibration affects the transformation matrix and, thus, leads to wrong Cartesian velocities.

Check the quality of the velocity data, as noise or errors in the beam data can propagate through the transformation. For instance, the Signal-to-Noise Ratio (SNR) should be generally larger than 20.






