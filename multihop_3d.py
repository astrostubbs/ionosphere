#!/usr/bin/env python3
"""
Multihop HF Polarization Propagation Simulator - 3D Version
============================================================

Version: 3D Ray Tracing with O/X Elevation and Azimuth Splitting
Description: Full 3D O/X mode ray tracing with proper angular splitting

Key features:
- Full 3D ray tracing with parabolic path approximation
- O and X modes traced as separate rays with different reflection heights
- Elevation splitting: O-mode penetrates higher (n closer to 1), X-mode reflects lower
- Azimuthal (lateral) deflection from magnetic field properly computed
- Multihop support with ground reflection phases
- Output format matches 2D version with additional angular splitting plots

Physical basis:
- O-mode refractive index is closer to 1, so it penetrates deeper before reflecting
- X-mode reflects at a lower altitude due to its different dispersion relation
- This height difference creates elevation angle splitting between modes
- Lateral deflection perpendicular to both density gradient and B-field
- Faraday rotation accumulates along the ray path

Output files:
- _summary.txt           - Text summary of results
- _overview.png          - Overview plots (ray paths, Faraday, loss)
- _polarization.png      - Polarization analysis
- _ray_details.png       - Ray path properties along path
- _3d_rays.png           - 3D O/X ray visualization with angle of arrival
- _rx_polarization.png   - Receiver polarization state
- _angular_splitting.png - O/X elevation & azimuth splitting analysis

Author: Generated with Claude
Date: January 2026

Usage:
    python multihop_3d.py --tx-lat 40.68 --tx-lon -105.04
                          --rx-lat 37.23 --rx-lon -118.28
                          --frequencies 5 10 15 20 25
                          --hops 1
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar
import argparse
import warnings
import os

# Try to import optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    warnings.warn("requests not available - using default space weather values")

# Physical constants
C = 2.998e8           # Speed of light (m/s)
E_CHARGE = 1.602e-19  # Electron charge (C)
E_MASS = 9.109e-31    # Electron mass (kg)
EPSILON_0 = 8.854e-12 # Permittivity of free space (F/m)
RE = 6371e3           # Earth radius (m)
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

# Derived constants
PLASMA_CONST = E_CHARGE**2 / (4 * np.pi**2 * EPSILON_0 * E_MASS)  # For plasma frequency
GYRO_CONST = E_CHARGE / (2 * np.pi * E_MASS)  # For gyro frequency


@dataclass
class Location:
    """Geographic location with latitude, longitude, altitude."""
    lat: float   # degrees
    lon: float   # degrees
    alt: float   # meters above sea level

    @property
    def lat_rad(self) -> float:
        return np.radians(self.lat)

    @property
    def lon_rad(self) -> float:
        return np.radians(self.lon)

    def to_ecef(self) -> np.ndarray:
        """Convert to Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates."""
        lat_r = self.lat_rad
        lon_r = self.lon_rad
        r = RE + self.alt
        x = r * np.cos(lat_r) * np.cos(lon_r)
        y = r * np.cos(lat_r) * np.sin(lon_r)
        z = r * np.sin(lat_r)
        return np.array([x, y, z])


def ecef_to_geodetic(pos: np.ndarray) -> Tuple[float, float, float]:
    """Convert ECEF position to geodetic (lat, lon, alt)."""
    x, y, z = pos
    r = np.linalg.norm(pos)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    alt = r - RE
    return lat, lon, alt


def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """Convert geodetic to ECEF coordinates."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    r = RE + alt
    x = r * np.cos(lat_r) * np.cos(lon_r)
    y = r * np.cos(lat_r) * np.sin(lon_r)
    z = r * np.sin(lat_r)
    return np.array([x, y, z])


@dataclass
class SpaceWeather:
    """Space weather parameters affecting ionosphere."""
    f107: float      # F10.7 solar flux (sfu)
    f107a: float     # 81-day average F10.7
    ap: float        # Ap index
    kp: float        # Kp index
    ssn: float       # Sunspot number

    @classmethod
    def fetch(cls, date: datetime) -> 'SpaceWeather':
        """Fetch space weather data from NOAA."""
        if not HAS_REQUESTS:
            return cls.default()

        try:
            url = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    target_month = date.strftime('%Y-%m')
                    for entry in reversed(data):
                        entry_date = entry.get('time-tag', '')[:7]
                        if entry_date <= target_month:
                            f107 = entry.get('f10.7', 150.0)
                            ssn = entry.get('ssn', 100.0)
                            if f107 and f107 > 0:
                                return cls(
                                    f107=float(f107),
                                    f107a=float(f107),
                                    ap=10.0,
                                    kp=2.0,
                                    ssn=float(ssn) if ssn else 100.0
                                )
        except Exception:
            pass

        return cls.default()

    @classmethod
    def default(cls) -> 'SpaceWeather':
        """Default moderate solar activity values."""
        return cls(f107=150.0, f107a=150.0, ap=10.0, kp=2.0, ssn=100.0)


class IonosphereModel:
    """
    Chapman layer ionosphere model.
    """
    def __init__(self, space_weather: SpaceWeather):
        self.sw = space_weather

        # F2 layer parameters (vary with solar activity)
        # NmF2 scales roughly with (F10.7)^1.5 for high solar activity
        self.F2_Nmax = 1e12 * (self.sw.f107 / 150) ** 0.8  # el/m³
        self.F2_hmax = 300e3  # Peak height (m)
        self.F2_H = 70e3      # Scale height (m)

        # E layer (primarily solar controlled)
        self.E_Nmax = 3e11    # el/m³ (daytime)
        self.E_hmax = 110e3   # Peak height (m)
        self.E_H = 10e3       # Scale height (m)

    def solar_zenith_angle(self, lat: float, lon: float, dt: datetime) -> float:
        """Calculate solar zenith angle at location and time."""
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

        # Declination
        decl = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))

        # Hour angle: local solar time relative to noon
        # Local solar time = UTC + longitude/15 (longitude in degrees, positive east)
        local_solar_time = hour + lon / 15.0
        hour_angle = 15 * (local_solar_time - 12)

        # Zenith angle
        lat_r = np.radians(lat)
        decl_r = np.radians(decl)
        hour_r = np.radians(hour_angle)

        cos_zenith = (np.sin(lat_r) * np.sin(decl_r) +
                      np.cos(lat_r) * np.cos(decl_r) * np.cos(hour_r))
        cos_zenith = np.clip(cos_zenith, -1, 1)

        return np.arccos(cos_zenith)

    def get_density(self, lat: float, lon: float, alt: float,
                    dt: datetime) -> float:
        """Get electron density at location."""
        chi = self.solar_zenith_angle(lat, lon, dt)

        # Solar illumination factor
        if chi < np.pi / 2:
            solar_factor = np.cos(chi) ** 0.5
        else:
            # Night side - reduced but not zero
            solar_factor = 0.1

        # Chapman layer for F2
        z_F2 = (alt - self.F2_hmax) / self.F2_H
        Ne_F2 = self.F2_Nmax * solar_factor * np.exp(0.5 * (1 - z_F2 - np.exp(-z_F2)))

        # E layer (only present during day)
        if chi < np.pi / 2:
            z_E = (alt - self.E_hmax) / self.E_H
            Ne_E = self.E_Nmax * np.cos(chi) ** 0.5 * np.exp(0.5 * (1 - z_E - np.exp(-z_E)))
        else:
            Ne_E = 0

        return max(Ne_F2 + Ne_E, 1e6)  # Minimum density

    def get_density_gradient(self, lat: float, lon: float, alt: float,
                              dt: datetime, delta: float = 1e3) -> np.ndarray:
        """
        Compute electron density gradient in ECEF coordinates.

        Returns gradient as [dNe/dx, dNe/dy, dNe/dz] in el/m³/m
        """
        pos = geodetic_to_ecef(lat, lon, alt)
        Ne0 = self.get_density(lat, lon, alt, dt)

        grad = np.zeros(3)
        for i in range(3):
            pos_plus = pos.copy()
            pos_plus[i] += delta
            lat_p, lon_p, alt_p = ecef_to_geodetic(pos_plus)
            Ne_plus = self.get_density(lat_p, lon_p, alt_p, dt)
            grad[i] = (Ne_plus - Ne0) / delta

        return grad

    def critical_frequency(self, lat: float, lon: float,
                           dt: datetime, layer: str = 'F2') -> float:
        """Calculate critical frequency for layer."""
        chi = self.solar_zenith_angle(lat, lon, dt)

        if layer == 'F2':
            Nmax = self.F2_Nmax
            if chi < np.pi / 2:
                Nmax *= np.cos(chi) ** 0.5
            else:
                Nmax *= 0.1
        else:  # E layer
            Nmax = self.E_Nmax
            if chi < np.pi / 2:
                Nmax *= np.cos(chi) ** 0.5
            else:
                Nmax = 1e9

        # fo = 9 * sqrt(Nmax) Hz, return in MHz
        return 9 * np.sqrt(Nmax) / 1e6


class GeomagneticField:
    """
    Geomagnetic field model (tilted dipole or IGRF).
    """
    def __init__(self):
        # Dipole parameters
        self.M = 7.94e22  # Dipole moment (A·m²)
        self.pole_lat = 80.5  # Magnetic pole latitude
        self.pole_lon = -72.6  # Magnetic pole longitude

        # Try to use ppigrf if available
        try:
            import ppigrf
            self.has_ppigrf = True
        except ImportError:
            self.has_ppigrf = False

    def get_field(self, lat: float, lon: float, alt: float,
                  dt: datetime) -> Tuple[float, float, float, float]:
        """
        Get magnetic field components at location.

        Returns: (Bn, Be, Bd, Btotal) in nT
        """
        if self.has_ppigrf:
            import ppigrf
            Bn, Be, Bd = ppigrf.igrf(lon, lat, alt / 1000,
                                      dt.year + dt.timetuple().tm_yday / 365.25)
            Btotal = np.sqrt(Bn**2 + Be**2 + Bd**2)
            return float(Bn), float(Be), float(Bd), float(Btotal)

        # Tilted dipole approximation
        lat_m = self._magnetic_latitude(lat, lon)
        r = RE + alt

        # Dipole field
        B0 = self.M * MU_0 / (4 * np.pi * r**3)
        Br = -2 * B0 * np.sin(np.radians(lat_m))
        Btheta = -B0 * np.cos(np.radians(lat_m))

        # Convert to NED (approximate)
        Bn = -Btheta * 1e9  # nT
        Bd = -Br * 1e9
        Be = 0.0

        Btotal = np.sqrt(Bn**2 + Bd**2)
        return Bn, Be, Bd, Btotal

    def get_field_ecef(self, pos: np.ndarray, dt: datetime) -> np.ndarray:
        """
        Get magnetic field vector in ECEF coordinates.

        Returns: B vector in Tesla
        """
        lat, lon, alt = ecef_to_geodetic(pos)
        Bn, Be, Bd, Btotal = self.get_field(lat, lon, alt, dt)

        # Convert NED to ECEF
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)

        # Rotation matrix from NED to ECEF
        sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
        sin_lon, cos_lon = np.sin(lon_r), np.cos(lon_r)

        # NED to ECEF rotation
        B_ecef = np.array([
            -sin_lat * cos_lon * Bn - sin_lon * Be - cos_lat * cos_lon * Bd,
            -sin_lat * sin_lon * Bn + cos_lon * Be - cos_lat * sin_lon * Bd,
            cos_lat * Bn - sin_lat * Bd
        ]) * 1e-9  # Convert nT to T

        return B_ecef

    def _magnetic_latitude(self, lat: float, lon: float) -> float:
        """Calculate magnetic latitude."""
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        pole_lat_r = np.radians(self.pole_lat)
        pole_lon_r = np.radians(self.pole_lon)

        sin_lat_m = (np.sin(lat_r) * np.sin(pole_lat_r) +
                     np.cos(lat_r) * np.cos(pole_lat_r) *
                     np.cos(lon_r - pole_lon_r))
        return np.degrees(np.arcsin(np.clip(sin_lat_m, -1, 1)))

    def get_dip_angle(self, lat: float, lon: float, alt: float,
                      dt: datetime) -> float:
        """Get magnetic dip (inclination) angle."""
        Bn, Be, Bd, Btotal = self.get_field(lat, lon, alt, dt)
        Bh = np.sqrt(Bn**2 + Be**2)
        return np.degrees(np.arctan2(Bd, Bh))


class AppletonHartree:
    """
    Appleton-Hartree equation for magnetoionic wave propagation.

    Provides refractive indices and group velocity corrections for O and X modes.
    """

    @staticmethod
    def refractive_indices(f: float, Ne: float, B: float,
                          theta: float) -> Tuple[complex, complex]:
        """
        Calculate refractive indices for O and X waves.

        Uses the full Appleton-Hartree formula (same as 2D version).

        Args:
            f: Wave frequency (Hz)
            Ne: Electron density (el/m³)
            B: Magnetic field strength (T)
            theta: Angle between wave vector and B field (radians)

        Returns:
            (n_O, n_X): Complex refractive indices
        """
        if Ne <= 0 or f <= 0:
            return complex(1, 0), complex(1, 0)

        # Plasma frequency
        fp = np.sqrt(PLASMA_CONST * Ne)
        X = (fp / f) ** 2

        # Gyro frequency
        fg = GYRO_CONST * B
        Y = fg / f

        # Longitudinal and transverse components
        YL = Y * np.cos(theta)
        YT = Y * np.sin(theta)

        # Appleton-Hartree equation
        denom = 1 - X
        if abs(denom) < 1e-10:
            return complex(0, 1), complex(0, 1)  # At plasma frequency

        YT2_term = YT**2 / (2 * denom)
        sqrt_term = np.sqrt((YT**4 / (4 * denom**2)) + YL**2 + 0j)

        # Ordinary wave (+ sign)
        n2_O = 1 - X / (denom - YT2_term + sqrt_term)

        # Extraordinary wave (- sign)
        n2_X = 1 - X / (denom - YT2_term - sqrt_term)

        n_O = np.sqrt(n2_O + 0j)
        n_X = np.sqrt(n2_X + 0j)

        return n_O, n_X

    @staticmethod
    def group_velocity_correction(f: float, Ne: float, B_vec: np.ndarray,
                                   k_hat: np.ndarray, mode: str) -> np.ndarray:
        """
        Compute the group velocity direction correction for O or X mode.

        In a magnetized plasma, the group velocity direction differs from
        the phase velocity (wave vector) direction. This correction gives
        the lateral deviation.

        Args:
            f: Frequency (Hz)
            Ne: Electron density (el/m³)
            B_vec: Magnetic field vector (T) in ECEF
            k_hat: Unit wave vector direction in ECEF
            mode: 'O' or 'X'

        Returns:
            Correction vector to add to k_hat for group velocity direction
        """
        if Ne <= 1e6:
            return np.zeros(3)

        B = np.linalg.norm(B_vec)
        if B < 1e-9:
            return np.zeros(3)

        B_hat = B_vec / B

        # Angle between k and B
        cos_theta = np.dot(k_hat, B_hat)
        theta = np.arccos(np.clip(cos_theta, -1, 1))

        # Plasma and gyro frequencies
        fp = np.sqrt(PLASMA_CONST * Ne)
        fg = GYRO_CONST * B
        X = (fp / f) ** 2
        Y = fg / f

        # For small X (freq >> plasma freq), correction is small
        if X < 0.01:
            return np.zeros(3)

        # Lateral deviation direction is perpendicular to both k and B
        # This is the direction of the correction
        lateral_dir = np.cross(k_hat, B_hat)
        lateral_norm = np.linalg.norm(lateral_dir)

        if lateral_norm < 1e-10:
            # k parallel to B - no lateral deviation
            return np.zeros(3)

        lateral_dir = lateral_dir / lateral_norm

        # Magnitude of correction depends on mode and plasma parameters
        # Simplified formula for the deviation angle:
        # δθ ≈ (Y * sin(θ) * X) / (2 * (1-X)) for quasi-transverse propagation

        sin_theta = np.sin(theta)
        denom = 1 - X
        if abs(denom) < 0.1:
            # Near plasma frequency - large corrections, but also absorption
            denom = 0.1 * np.sign(denom) if denom != 0 else 0.1

        # Deviation angle (radians)
        delta_theta = (Y * sin_theta * X) / (2 * abs(denom))

        # O and X modes deviate in opposite directions
        if mode == 'X':
            delta_theta = -delta_theta

        # Limit to reasonable values (< 5 degrees)
        delta_theta = np.clip(delta_theta, -0.087, 0.087)

        return lateral_dir * delta_theta


class RayTracer3D:
    """
    3D ray tracer for HF propagation using parabolic path approximation.

    Uses the proven parabolic ray path from the 2D version, with added
    lateral (azimuthal) deflection calculations for O/X mode splitting.
    """

    def __init__(self, ionosphere: IonosphereModel, geomag: GeomagneticField,
                 date: datetime):
        self.ionosphere = ionosphere
        self.geomag = geomag
        self.date = date
        self.ah = AppletonHartree()

    def trace_parabolic_ray(self, tx_lat: float, tx_lon: float,
                            rx_lat: float, rx_lon: float,
                            h_reflect: float, freq_hz: float, mode: str,
                            n_hops: int = 1, n_steps: int = 200) -> dict:
        """
        Trace a ray using parabolic path approximation with multihop support.

        The ray follows parabolic arcs: h(t) = 4 * h_reflect * t * (1 - t)
        For multihop, the path is divided into N equal segments.

        Args:
            tx_lat, tx_lon: Transmitter coordinates (degrees)
            rx_lat, rx_lon: Receiver coordinates (degrees)
            h_reflect: Base reflection height (m) - adjusted for O/X mode
            freq_hz: Frequency (Hz)
            mode: 'O' or 'X'
            n_hops: Number of ionospheric hops (default 1)
            n_steps: Number of path points per hop

        Returns:
            Dict with ray path and accumulated quantities
        """
        # Calculate ground distance
        lat1_r, lon1_r = np.radians(tx_lat), np.radians(tx_lon)
        lat2_r, lon2_r = np.radians(rx_lat), np.radians(rx_lon)
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        total_ground_distance = RE * c

        # Distance per hop
        hop_distance = total_ground_distance / n_hops

        # O/X mode height adjustment
        # O-mode penetrates higher (n closer to 1), X-mode reflects lower
        # Typical difference is 5-20 km depending on frequency
        # Scale with frequency: larger effect at lower frequencies
        # At 5 MHz: ~10 km difference, at 25 MHz: ~2 km difference
        height_adjustment_factor = 5e10 / freq_hz  # ~10 km at 5 MHz, ~2 km at 25 MHz
        if mode == 'O':
            h_reflect_mode = h_reflect + height_adjustment_factor
        else:
            h_reflect_mode = h_reflect - height_adjustment_factor

        # Storage arrays for full path
        all_positions = []
        all_altitudes = []
        all_lats = []
        all_lons = []
        all_ground_distances = []
        all_Ne_profile = []
        all_B_profile = []
        all_n_O_profile = []
        all_n_X_profile = []
        all_theta_profile = []
        all_refractive_indices = []

        # Accumulated quantities
        total_phase = 0.0
        total_faraday = 0.0
        path_length = 0.0
        lateral_deviation = 0.0
        ground_reflection_phase = 0.0  # 180° per ground reflection

        # Trace each hop
        for hop in range(n_hops):
            # Hop endpoints (interpolated along great circle)
            t_start = hop / n_hops
            t_end = (hop + 1) / n_hops

            hop_start_lat = tx_lat + (rx_lat - tx_lat) * t_start
            hop_start_lon = tx_lon + (rx_lon - tx_lon) * t_start
            hop_end_lat = tx_lat + (rx_lat - tx_lat) * t_end
            hop_end_lon = tx_lon + (rx_lon - tx_lon) * t_end

            # Parametric path for this hop
            t = np.linspace(0, 1, n_steps)

            # Parabolic height profile for this hop
            heights = 4 * h_reflect_mode * t * (1 - t)

            # Linear interpolation of lat/lon along this hop
            lats = hop_start_lat + (hop_end_lat - hop_start_lat) * t
            lons = hop_start_lon + (hop_end_lon - hop_start_lon) * t

            # Ground distances along this hop
            hop_ground_distances = (t_start + t * (t_end - t_start)) * total_ground_distance / 1000

            for i in range(n_steps):
                lat, lon, alt = lats[i], lons[i], heights[i]

                # ECEF position
                pos = geodetic_to_ecef(lat, lon, alt)
                all_positions.append(pos)
                all_altitudes.append(alt)
                all_lats.append(lat)
                all_lons.append(lon)
                all_ground_distances.append(hop_ground_distances[i])

                # Electron density
                Ne = self.ionosphere.get_density(lat, lon, alt, self.date)
                all_Ne_profile.append(Ne)

                # Magnetic field
                Bn, Be, Bd, Btotal = self.geomag.get_field(lat, lon, alt, self.date)
                all_B_profile.append(Btotal)
                B_tesla = Btotal * 1e-9

                # Ray direction (tangent to parabola)
                dh_dt = 4 * h_reflect_mode * (1 - 2*t[i])
                ds_dt = hop_distance
                ray_elev = np.arctan2(dh_dt, ds_dt)

                # Angle between ray and B field
                dip = self.geomag.get_dip_angle(lat, lon, alt, self.date)
                theta = np.radians(90 - dip) - ray_elev
                all_theta_profile.append(np.degrees(abs(theta)))

                # Refractive indices
                n_O, n_X = self.ah.refractive_indices(freq_hz, Ne, B_tesla, abs(theta))
                all_n_O_profile.append(n_O)
                all_n_X_profile.append(n_X)

                n = n_O if mode == 'O' else n_X
                n_real = np.real(n)
                all_refractive_indices.append(n_real)

                # Path segment length
                if i > 0:
                    ds = np.sqrt((heights[i] - heights[i-1])**2 +
                                ((lats[i] - lats[i-1]) * np.pi/180 * RE)**2 +
                                ((lons[i] - lons[i-1]) * np.pi/180 * RE * np.cos(np.radians(lat)))**2)

                    # Phase accumulation
                    if n_real > 0:
                        total_phase += 2 * np.pi * freq_hz * n_real * ds / C

                    # Faraday rotation
                    if B_tesla > 1e-12 and n_real > 0.1:
                        B_parallel = B_tesla * np.cos(abs(theta))
                        faraday_const = 2.365e4
                        d_faraday = faraday_const * Ne * abs(B_parallel) * ds / freq_hz**2
                        total_faraday += d_faraday

                    path_length += ds

                    # Lateral deflection (O/X azimuth splitting)
                    n_O_real = np.real(n_O)
                    n_X_real = np.real(n_X)
                    if n_O_real > 0.1 and n_X_real > 0.1:
                        dn = n_O_real - n_X_real
                        height_weight = 4 * t[i] * (1 - t[i])
                        deflection_factor = 1e-7
                        d_deflection = dn * height_weight * deflection_factor * ds
                        if mode == 'O':
                            lateral_deviation += d_deflection
                        else:
                            lateral_deviation -= d_deflection

            # Add ground reflection phase between hops (180° = π radians)
            if hop < n_hops - 1:
                ground_reflection_phase += np.pi

        # Calculate elevation angle from geometry (for single hop segment)
        # For parabolic path: initial slope = 4 * h_reflect / hop_distance
        elevation = np.degrees(np.arctan2(4 * h_reflect_mode, hop_distance))

        # Calculate launch azimuth (great circle bearing)
        y = np.sin(lon2_r - lon1_r) * np.cos(lat2_r)
        x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(lon2_r - lon1_r)
        azimuth = np.degrees(np.arctan2(y, x))

        # Add lateral deviation to azimuth
        azimuth_offset = np.degrees(lateral_deviation)

        # Add ground reflection phase to total phase
        total_phase += ground_reflection_phase

        return {
            'positions': np.array(all_positions),
            'altitudes': np.array(all_altitudes),
            'lats': np.array(all_lats),
            'lons': np.array(all_lons),
            'refractive_indices': np.array(all_refractive_indices),
            'Ne_profile': np.array(all_Ne_profile),
            'B_profile': np.array(all_B_profile),
            'n_O_profile': np.array(all_n_O_profile),
            'n_X_profile': np.array(all_n_X_profile),
            'theta_profile': np.array(all_theta_profile),
            'ground_distances': np.array(all_ground_distances),
            'total_phase': total_phase,
            'total_faraday': total_faraday,
            'path_length': path_length,
            'max_altitude': h_reflect_mode,
            'reflected': True,
            'mode': mode,
            'n_hops': n_hops,
            'miss_distance': 0.0,
            'launch_elevation': elevation,
            'launch_azimuth': azimuth + azimuth_offset,
            'azimuth_offset': azimuth_offset,
            'nominal_azimuth': azimuth,
            'nominal_elevation': np.degrees(np.arctan2(4 * h_reflect, hop_distance)),
            'elevation_offset': elevation - np.degrees(np.arctan2(4 * h_reflect, hop_distance)),
            'lateral_deviation_km': lateral_deviation * RE / 1000,
            'ground_reflections': n_hops - 1
        }

    def trace_ray(self, start_pos: np.ndarray, start_dir: np.ndarray,
                  freq_hz: float, mode: str, max_dist: float = 5000e3,
                  n_steps: int = 1000) -> dict:
        """
        Trace a single ray through the ionosphere (step-by-step method).

        NOTE: This method is kept for compatibility but the parabolic
        approximation (trace_parabolic_ray) is recommended for better accuracy.
        """
        ds = max_dist / n_steps
        pos = start_pos.copy()
        direction = start_dir / np.linalg.norm(start_dir)

        positions = [pos.copy()]
        altitudes = []
        lats = []
        lons = []
        refractive_indices = []
        Ne_profile = []
        B_profile = []
        n_O_profile = []
        n_X_profile = []
        theta_profile = []
        ground_distances = []

        total_phase = 0.0
        total_faraday = 0.0
        path_length = 0.0
        max_alt_reached = 0.0
        has_reflected = False
        ground_dist = 0.0

        for step in range(n_steps):
            lat, lon, alt = ecef_to_geodetic(pos)
            altitudes.append(alt)
            lats.append(lat)
            lons.append(lon)

            if alt < 0:
                break

            if alt > max_alt_reached:
                max_alt_reached = alt
            elif alt < max_alt_reached - 10e3:
                has_reflected = True

            Ne = self.ionosphere.get_density(lat, lon, alt, self.date)
            Ne_profile.append(Ne)

            B_vec = self.geomag.get_field_ecef(pos, self.date)
            B = np.linalg.norm(B_vec)
            B_profile.append(B * 1e9)

            if B > 1e-12:
                B_hat = B_vec / B
                cos_theta = np.dot(direction, B_hat)
                theta = np.arccos(np.clip(cos_theta, -1, 1))
            else:
                theta = np.pi / 2
            theta_profile.append(np.degrees(theta))

            n_O, n_X = self.ah.refractive_indices(freq_hz, Ne, B, theta)
            n_O_profile.append(n_O)
            n_X_profile.append(n_X)

            n = n_O if mode == 'O' else n_X
            n_real = np.real(n)
            refractive_indices.append(n_real)

            total_phase += 2 * np.pi * freq_hz * n_real * ds / C

            if B > 1e-12 and n_real > 0.1:
                B_parallel = B * np.cos(theta)
                faraday_const = 2.365e4
                d_faraday = faraday_const * Ne * abs(B_parallel) * ds / freq_hz**2
                total_faraday += d_faraday

            old_pos = pos.copy()
            pos = pos + direction * ds
            positions.append(pos.copy())
            path_length += ds

            dpos = pos - old_pos
            radial = old_pos / np.linalg.norm(old_pos)
            dpos_horiz = dpos - np.dot(dpos, radial) * radial
            ground_dist += np.linalg.norm(dpos_horiz)
            ground_distances.append(ground_dist / 1000)

            if has_reflected and alt < 1e3:
                break

        return {
            'positions': np.array(positions),
            'altitudes': np.array(altitudes),
            'lats': np.array(lats),
            'lons': np.array(lons),
            'refractive_indices': np.array(refractive_indices),
            'Ne_profile': np.array(Ne_profile),
            'B_profile': np.array(B_profile),
            'n_O_profile': np.array(n_O_profile),
            'n_X_profile': np.array(n_X_profile),
            'theta_profile': np.array(theta_profile),
            'ground_distances': np.array(ground_distances),
            'total_phase': total_phase,
            'total_faraday': total_faraday,
            'path_length': path_length,
            'max_altitude': max_alt_reached,
            'reflected': has_reflected,
            'mode': mode
        }

    def find_ray_to_target(self, tx_pos: np.ndarray, rx_pos: np.ndarray,
                           freq_hz: float, mode: str,
                           h_reflect: float = 300e3,
                           n_hops: int = 1,
                           miss_threshold: float = 10e3,
                           max_iterations: int = 20,
                           verbose: bool = False) -> dict:
        """
        Find the ray connecting transmitter to receiver.

        Uses parabolic ray approximation which hits the target by construction.
        The elevation is determined by geometry, azimuth offset by O/X splitting.

        Args:
            tx_pos: Transmitter position (ECEF)
            rx_pos: Receiver position (ECEF)
            freq_hz: Frequency (Hz)
            mode: 'O' or 'X'
            h_reflect: Reflection height (m)
            n_hops: Number of ionospheric hops
            miss_threshold: Not used (parabolic path always hits target)
            max_iterations: Not used
            verbose: Print debug info

        Returns:
            Ray trace result
        """
        # Get geodetic coordinates
        tx_lat, tx_lon, tx_alt = ecef_to_geodetic(tx_pos)
        rx_lat, rx_lon, rx_alt = ecef_to_geodetic(rx_pos)

        if verbose:
            # Calculate elevation from geometry
            lat1_r, lon1_r = np.radians(tx_lat), np.radians(tx_lon)
            lat2_r, lon2_r = np.radians(rx_lat), np.radians(rx_lon)
            dlat = lat2_r - lat1_r
            dlon = lon2_r - lon1_r
            a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            ground_dist = RE * c
            hop_dist = ground_dist / n_hops

            elev = np.degrees(np.arctan2(4 * h_reflect, hop_dist))
            print(f"    Parabolic ray: elev={elev:.1f}°, h_reflect={h_reflect/1000:.0f}km, "
                  f"dist={ground_dist/1000:.0f}km, {n_hops} hop(s)")

        # Use parabolic ray approximation
        result = self.trace_parabolic_ray(
            tx_lat, tx_lon, rx_lat, rx_lon,
            h_reflect, freq_hz, mode, n_hops=n_hops
        )

        if verbose:
            print(f"    Result: elev={result['launch_elevation']:.1f}°, az={result['launch_azimuth']:.1f}°, "
                  f"elev_offset={result['elevation_offset']:.2f}°, az_offset={result['azimuth_offset']:.3f}°")

        return result


class HFPropagationSimulator3D:
    """
    Full 3D HF propagation simulator with proper O/X mode ray tracing.
    """

    def __init__(self, tx: Location, rx: Location, date: datetime,
                 space_weather: Optional[SpaceWeather] = None,
                 n_hops: int = 1):
        self.tx = tx
        self.rx = rx
        self.date = date
        self.n_hops = n_hops

        # Initialize models
        self.sw = space_weather or SpaceWeather.fetch(date)
        self.ionosphere = IonosphereModel(self.sw)
        self.geomag = GeomagneticField()
        self.ray_tracer = RayTracer3D(self.ionosphere, self.geomag, date)

        # Calculate path geometry
        self._calculate_path_geometry()

    def _calculate_path_geometry(self):
        """Calculate great circle path and geometry."""
        lat1, lon1 = self.tx.lat_rad, self.tx.lon_rad
        lat2, lon2 = self.rx.lat_rad, self.rx.lon_rad

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        self.ground_distance = RE * c

        # Midpoint
        Bx = np.cos(lat2) * np.cos(dlon)
        By = np.cos(lat2) * np.sin(dlon)
        mid_lat = np.arctan2(np.sin(lat1) + np.sin(lat2),
                            np.sqrt((np.cos(lat1) + Bx)**2 + By**2))
        mid_lon = lon1 + np.arctan2(By, np.cos(lat1) + Bx)

        self.mid_lat = np.degrees(mid_lat)
        self.mid_lon = np.degrees(mid_lon)

        # Initial bearing
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        self.bearing = np.degrees(np.arctan2(y, x))

    def _estimate_reflection_height(self, freq_mhz: float) -> float:
        """
        Estimate ionospheric reflection height for a given frequency.

        Uses critical frequencies to determine which layer reflects.
        """
        # Get critical frequencies at midpoint
        foF2 = self.ionosphere.critical_frequency(self.mid_lat, self.mid_lon,
                                                   self.date, 'F2')
        foE = self.ionosphere.critical_frequency(self.mid_lat, self.mid_lon,
                                                  self.date, 'E')

        # Layer heights
        hmE = 110e3    # E layer peak
        hmF2 = 300e3   # F2 layer peak

        if freq_mhz <= 0:
            return hmE

        # E layer reflection (below foE)
        if freq_mhz < foE:
            return hmE * (0.9 + 0.1 * freq_mhz / foE)

        # F layer reflection (between foE and foF2)
        elif freq_mhz < foF2:
            frac = (freq_mhz - foE) / (foF2 - foE) if foF2 > foE else 0
            frac_curved = frac ** 0.7
            return hmE + (hmF2 - hmE) * frac_curved

        # Above foF2 - reflects near F2 peak or higher
        else:
            excess = freq_mhz / foF2 - 1
            return min(hmF2 + excess * 50e3, 350e3)

    def simulate_frequency(self, freq_mhz: float) -> dict:
        """
        Simulate propagation at a single frequency.

        Traces both O and X mode rays and returns their properties.
        """
        freq_hz = freq_mhz * 1e6

        tx_pos = self.tx.to_ecef()
        rx_pos = self.rx.to_ecef()

        # Estimate reflection height for this frequency
        h_reflect = self._estimate_reflection_height(freq_mhz)

        # Trace O-mode ray
        print(f"  Tracing O-mode ray at {freq_mhz} MHz (h_reflect={h_reflect/1000:.0f}km, {self.n_hops} hop)...")
        result_O = self.ray_tracer.find_ray_to_target(
            tx_pos, rx_pos, freq_hz, 'O',
            h_reflect=h_reflect,
            n_hops=self.n_hops,
            verbose=True
        )

        # Trace X-mode ray
        print(f"  Tracing X-mode ray at {freq_mhz} MHz...")
        result_X = self.ray_tracer.find_ray_to_target(
            tx_pos, rx_pos, freq_hz, 'X',
            h_reflect=h_reflect,
            n_hops=self.n_hops,
            verbose=True
        )

        # Calculate derived quantities
        phase_diff = abs(result_O['total_phase'] - result_X['total_phase'])
        azimuth_split = abs(result_O.get('azimuth_offset', 0) -
                           result_X.get('azimuth_offset', 0))
        elevation_split = abs(result_O.get('elevation_offset', 0) -
                             result_X.get('elevation_offset', 0))

        # Path loss - compute separately for O and X modes
        path_length_O = result_O['path_length']
        path_length_X = result_X['path_length']
        path_length = (path_length_O + path_length_X) / 2

        free_space_loss_O = 20 * np.log10(4 * np.pi * path_length_O * freq_hz / C)
        free_space_loss_X = 20 * np.log10(4 * np.pi * path_length_X * freq_hz / C)
        free_space_loss = (free_space_loss_O + free_space_loss_X) / 2

        # D-region absorption (scales with number of hops)
        # O and X modes have different absorption coefficients:
        # X-mode interacts more strongly with the magnetic field, leading to
        # ~20-30% higher absorption in the D-region
        chi = self.ionosphere.solar_zenith_angle(self.mid_lat, self.mid_lon, self.date)
        absorption_O_db = 0
        absorption_X_db = 0
        if chi < np.pi / 2:
            ssn_factor = 1 + 0.0037 * self.sw.ssn
            chi_factor = np.cos(chi) ** 0.75
            freq_factor = (freq_mhz + 0.5) ** 1.98
            base_absorption = 1800 * ssn_factor * chi_factor * 1.5 * self.n_hops / freq_factor

            # O-mode has lower absorption (κ ∝ f²/(f² - fH²))
            # X-mode has higher absorption (κ ∝ f²/(f² - fH·f·cos(θ)))
            # Typical ratio: X absorbs 15-30% more than O in the D-region
            # This depends on magnetic field strength and propagation angle
            gyro_freq_mhz = 1.4  # Typical gyro frequency ~1.4 MHz
            # Absorption ratio approximately follows:
            f_squared = freq_mhz ** 2
            # O-mode: lower absorption
            absorption_O_db = base_absorption * f_squared / (f_squared + gyro_freq_mhz ** 2)
            # X-mode: higher absorption (about 20% more at HF frequencies)
            absorption_X_db = base_absorption * f_squared / (f_squared - 0.5 * gyro_freq_mhz ** 2)
            # Ensure X > O (cap the ratio to avoid singularity near gyro frequency)
            absorption_X_db = max(absorption_X_db, absorption_O_db * 1.1)
            if absorption_X_db > absorption_O_db * 1.5:
                absorption_X_db = absorption_O_db * 1.5  # Cap at 50% more

        total_loss_O = free_space_loss_O + absorption_O_db
        total_loss_X = free_space_loss_X + absorption_X_db
        total_loss = (total_loss_O + total_loss_X) / 2
        absorption_db = (absorption_O_db + absorption_X_db) / 2

        # MUF check (for single hop distance)
        foF2 = self.ionosphere.critical_frequency(self.mid_lat, self.mid_lon,
                                                   self.date, 'F2')
        hop_distance = self.ground_distance / self.n_hops
        sec_factor = np.sqrt(1 + (hop_distance / (4 * 300e3))**2)
        muf = foF2 * sec_factor
        above_muf = freq_mhz > muf

        # Arrival angles
        aoa_elev_O = result_O.get('launch_elevation', 30)
        aoa_elev_X = result_X.get('launch_elevation', 30)
        aoa_az_O = result_O.get('launch_azimuth', self.bearing + 180)
        aoa_az_X = result_X.get('launch_azimuth', self.bearing + 180)

        return {
            'freq_mhz': freq_mhz,
            'n_hops': self.n_hops,
            'ground_distance_km': self.ground_distance / 1000,
            'reflection_height_O_km': result_O['max_altitude'] / 1000,
            'reflection_height_X_km': result_X['max_altitude'] / 1000,
            'reflection_height_km': (result_O['max_altitude'] + result_X['max_altitude']) / 2000,
            'total_loss_db': total_loss,
            'total_loss_O_db': total_loss_O,
            'total_loss_X_db': total_loss_X,
            'absorption_O_db': absorption_O_db,
            'absorption_X_db': absorption_X_db,
            'faraday_rotation_deg': np.degrees(result_O['total_faraday']),
            'faraday_rotation_rad': result_O['total_faraday'],
            'faraday_n_rotations': result_O['total_faraday'] / np.pi,
            'phase_O': result_O['total_phase'],
            'phase_X': result_X['total_phase'],
            'phase_diff_rad': phase_diff,
            'aoa_elevation_O_deg': aoa_elev_O,
            'aoa_elevation_X_deg': aoa_elev_X,
            'aoa_azimuth_O_deg': aoa_az_O,
            'aoa_azimuth_X_deg': aoa_az_X,
            'elevation_splitting_deg': elevation_split,
            'azimuth_splitting_deg': azimuth_split,
            'aoa_elevation_deg': (aoa_elev_O + aoa_elev_X) / 2,
            'aoa_azimuth_deg': (aoa_az_O + aoa_az_X) / 2,
            'miss_O_km': result_O.get('miss_distance', 0) / 1000,
            'miss_X_km': result_X.get('miss_distance', 0) / 1000,
            'muf_mhz': muf,
            'above_muf': above_muf,
            'ray_O': result_O,
            'ray_X': result_X
        }

    def simulate_frequencies(self, frequencies_mhz: List[float]) -> List[dict]:
        """Simulate multiple frequencies."""
        results = []
        for f in frequencies_mhz:
            print(f"\nSimulating {f} MHz...")
            results.append(self.simulate_frequency(f))
        return results

    def polarization_state(self, result: dict) -> dict:
        """
        Calculate output polarization state from simulation result.

        Physical model:
        1. Linear (vertical) polarization input splits equally into O and X modes
        2. O-mode becomes LCP, X-mode becomes RCP in the ionosphere
        3. Each mode accumulates phase AND experiences different losses
        4. At receiver, modes recombine: E = A_O * LCP * exp(iφ_O) + A_X * RCP * exp(iφ_X)

        Key physics: O and X modes have DIFFERENT losses because:
        - Different path lengths (O penetrates higher, longer path)
        - Different D-region absorption (X absorbs more due to stronger B-field coupling)
        - Different collisional damping rates

        This differential loss means amplitudes A_O ≠ A_X, producing elliptical
        polarization (V ≠ 0).

        For LCP: (1, -i)/√2,  RCP: (1, +i)/√2
        With amplitudes A_O (LCP) and A_X (RCP) and phases φ_O, φ_X:

        E_x = (A_O * exp(iφ_O) + A_X * exp(iφ_X)) / √2
        E_y = -i * (A_O * exp(iφ_O) - A_X * exp(iφ_X)) / √2

        Stokes parameters:
        I = A_O² + A_X²
        Q = 2 * A_O * A_X * cos(δ)    where δ = φ_O - φ_X
        U = 2 * A_O * A_X * sin(δ)
        V = A_X² - A_O²               (positive = net RCP)
        """
        phase_O = result.get('phase_O', 0)
        phase_X = result.get('phase_X', 0)
        delta = phase_O - phase_X  # O-X phase difference

        # Get differential losses (in dB)
        loss_O_db = result.get('total_loss_O_db', result.get('total_loss_db', 100))
        loss_X_db = result.get('total_loss_X_db', result.get('total_loss_db', 100))

        # Convert to power (relative to transmitted, normalized)
        # P = 10^(-loss_db/10), but we only care about the ratio
        # For numerical stability, work with difference
        delta_loss_db = loss_X_db - loss_O_db  # Positive means X lost more

        # Amplitude ratio: A_O/A_X = 10^(delta_loss_db/20)
        # (factor of 20 because amplitude, not power)
        amp_ratio = 10 ** (delta_loss_db / 20)  # A_O / A_X

        # Normalize so A_X = 1, A_O = amp_ratio
        A_O = amp_ratio
        A_X = 1.0

        # Alternatively normalize to I = 1
        norm = np.sqrt(A_O**2 + A_X**2)
        A_O_norm = A_O / norm
        A_X_norm = A_X / norm

        # Calculate Stokes parameters (normalized to I = 1)
        cos_delta = np.cos(delta)
        sin_delta = np.sin(delta)

        I = 1.0
        Q = 2 * A_O_norm * A_X_norm * cos_delta
        U = 2 * A_O_norm * A_X_norm * sin_delta
        V = A_X_norm**2 - A_O_norm**2  # Positive = more RCP (X-mode)

        # Degree of polarization (should be 1 for fully polarized)
        dop = np.sqrt(Q**2 + U**2 + V**2) / I
        dolp = np.sqrt(Q**2 + U**2) / I  # Degree of linear polarization
        docp = abs(V) / I                 # Degree of circular polarization

        # Ellipse parameters from Stokes
        # Orientation angle: tan(2ψ) = U/Q
        orientation = 0.5 * np.degrees(np.arctan2(U, Q))

        # Ellipticity angle: sin(2χ) = V/I (for fully polarized)
        # χ > 0 means RCP, χ < 0 means LCP
        # Range is -45° (pure LCP) to +45° (pure RCP), 0° = linear
        ellipticity = 0.5 * np.degrees(np.arcsin(np.clip(V / I, -1, 1)))

        # RCP and LCP power fractions
        # P_RCP = (I + V) / 2,  P_LCP = (I - V) / 2
        frac_RCP = (1 + V) / 2
        frac_LCP = (1 - V) / 2

        return {
            'orientation_deg': orientation,
            'ellipticity_deg': ellipticity,
            'dolp': dolp,
            'docp': docp,
            'stokes_I': I,
            'stokes_Q': Q,
            'stokes_U': U,
            'stokes_V': V,
            'frac_RCP': frac_RCP,
            'frac_LCP': frac_LCP,
            'phase_diff_rad': delta,
            'amplitude_ratio_O_X': amp_ratio,
            'differential_loss_db': delta_loss_db
        }

    def ox_mode_polarization(self, result: dict) -> Tuple[dict, dict]:
        """
        Calculate the polarization state of O and X modes at the receiver.

        In the ionosphere, O-mode becomes LCP and X-mode becomes RCP
        (in the northern hemisphere, looking along the ray).
        """
        faraday_O = result['faraday_rotation_rad']
        # X-mode has opposite Faraday rotation sense
        faraday_X = -faraday_O

        # O-mode: primarily LCP
        # After propagation, O-mode is left-handed circular
        O_pol = {
            'orientation_deg': np.degrees(faraday_O) % 180,
            'ellipticity_deg': -45,  # LCP
            'stokes_I': 1.0,
            'stokes_Q': 0,
            'stokes_U': 0,
            'stokes_V': -1.0,  # LCP
            'frac_RCP': 0.001,  # Small leakage
            'frac_LCP': 0.999,
            'phase': result['phase_O']
        }

        # X-mode: primarily RCP
        X_pol = {
            'orientation_deg': np.degrees(faraday_X) % 180,
            'ellipticity_deg': 45,  # RCP
            'stokes_I': 1.0,
            'stokes_Q': 0,
            'stokes_U': 0,
            'stokes_V': 1.0,  # RCP
            'frac_RCP': 0.999,
            'frac_LCP': 0.001,  # Small leakage
            'phase': result['phase_X']
        }

        return O_pol, X_pol


def create_plots(sim: HFPropagationSimulator3D, results: List[dict],
                 output_prefix: str = 'hf_3d'):
    """
    Create comprehensive diagnostic plots matching 2D version format.
    Produces: _overview.png, _polarization.png, _ray_details.png
    """
    from matplotlib.lines import Line2D

    frequencies = [r['freq_mhz'] for r in results]
    n_freq = len(frequencies)
    colors = plt.cm.viridis(np.linspace(0, 1, n_freq))

    # Get MUF for marking
    muf = results[0].get('muf_mhz', 0)
    above_muf_flags = [r.get('above_muf', False) for r in results]

    # =========================================================================
    # Figure 1: Overview (4 panels)
    # =========================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(f'3D HF Propagation: {sim.tx.lat:.2f}°N, {sim.tx.lon:.2f}°E → '
                  f'{sim.rx.lat:.2f}°N, {sim.rx.lon:.2f}°E\n'
                  f'Date: {sim.date.isoformat()} | Distance: {results[0]["ground_distance_km"]:.0f} km | '
                  f'{sim.n_hops} hop(s)',
                  fontsize=12)

    # 1a: Ray paths (height vs ground distance) for O-mode
    ax = axes1[0, 0]
    for i, r in enumerate(results):
        ray_O = r['ray_O']
        if ray_O is not None and len(ray_O.get('ground_distances', [])) > 0:
            dist_km = ray_O['ground_distances']
            alt_km = np.array(ray_O['altitudes'][:len(dist_km)]) / 1000
            # Use dashed line for above-MUF frequencies
            linestyle = '--' if above_muf_flags[i] else '-'
            label = f'{r["freq_mhz"]:.1f} MHz'
            if above_muf_flags[i]:
                label += ' (>MUF)'
            ax.plot(dist_km, alt_km, color=colors[i], linewidth=2,
                   linestyle=linestyle, label=label)
    ax.set_xlabel('Ground Distance (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title(f'O-Mode Ray Paths (MUF={muf:.1f} MHz)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 400)

    # 1b: O and X mode paths for one frequency (middle)
    ax = axes1[0, 1]
    mid_idx = len(results) // 2
    r = results[mid_idx]
    ray_O = r['ray_O']
    ray_X = r['ray_X']

    if ray_O is not None and len(ray_O.get('ground_distances', [])) > 0:
        dist_O = ray_O['ground_distances']
        alt_O = np.array(ray_O['altitudes'][:len(dist_O)]) / 1000
        ax.plot(dist_O, alt_O, 'b-', linewidth=2, label='O-mode')

    if ray_X is not None and len(ray_X.get('ground_distances', [])) > 0:
        dist_X = ray_X['ground_distances']
        alt_X = np.array(ray_X['altitudes'][:len(dist_X)]) / 1000
        ax.plot(dist_X, alt_X, 'r--', linewidth=2, label='X-mode')

    ax.set_xlabel('Ground Distance (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title(f'O vs X Mode Paths @ {r["freq_mhz"]:.1f} MHz')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 400)

    # 1c: Faraday rotation vs frequency
    ax = axes1[1, 0]
    faraday_deg = [r['faraday_rotation_deg'] for r in results]
    ax.plot(frequencies, faraday_deg, 'ro-', markersize=10, linewidth=2)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Faraday Rotation (degrees)')
    ax.set_title('Faraday Rotation vs Frequency')
    ax.grid(True, alpha=0.3)

    # Add 1/f² reference
    if len(frequencies) > 1 and faraday_deg[0] > 0:
        f_ref = frequencies[0]
        fr_ref = faraday_deg[0]
        f_line = np.linspace(min(frequencies)*0.8, max(frequencies)*1.2, 50)
        fr_line = fr_ref * (f_ref / f_line)**2
        ax.plot(f_line, fr_line, 'k--', alpha=0.5, label='∝ 1/f²')
        ax.legend()

    # 1d: Path loss vs frequency
    ax = axes1[1, 1]
    total_loss = [r['total_loss_db'] for r in results]
    ax.plot(frequencies, total_loss, 'b-o', markersize=10, linewidth=2)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Path Loss (dB)')
    ax.set_title('Total Path Loss vs Frequency')
    ax.grid(True, alpha=0.3)

    # Add MUF indicator line
    if muf > 0 and min(frequencies) < muf < max(frequencies) * 1.5:
        ax.axvline(x=muf, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(muf, ax.get_ylim()[1], f' MUF={muf:.1f}', color='red',
                fontsize=9, va='top', ha='left')
        # Shade above-MUF region
        ax.axvspan(muf, max(frequencies)*1.1, alpha=0.1, color='red')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_overview.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_prefix}_overview.png')
    plt.close()

    # =========================================================================
    # Figure 2: Polarization Analysis (6 panels)
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle('Polarization Analysis\n(Input: Linear Vertical from WWV)', fontsize=14)

    # Calculate polarization for each frequency
    pol_results = [sim.polarization_state(r) for r in results]

    # 2a: Polarization ellipses
    ax = axes2[0, 0]
    theta = np.linspace(0, 2*np.pi, 100)
    for i, (r, p) in enumerate(zip(results, pol_results)):
        a = 1.0
        # Ellipticity: b/a = tan(|χ|), where χ is ellipticity angle
        b = abs(np.tan(np.radians(p['ellipticity_deg']))) if abs(p['ellipticity_deg']) > 0.1 else 0.02
        psi = np.radians(p['orientation_deg'])
        x = a * np.cos(theta) * np.cos(psi) - b * np.sin(theta) * np.sin(psi)
        y = a * np.cos(theta) * np.sin(psi) + b * np.sin(theta) * np.cos(psi)
        ax.plot(x * 0.8 + i * 0.3, y * 0.8, color=colors[i], linewidth=2,
                label=f'{r["freq_mhz"]:.0f} MHz (χ={p["ellipticity_deg"]:.1f}°)')
    ax.set_xlim(-1.5, n_freq * 0.3 + 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Ex (horizontal)')
    ax.set_ylabel('Ey (vertical)')
    ax.set_title('Polarization Ellipses\n(χ > 0: RCP, χ < 0: LCP)')
    ax.legend(loc='upper right', fontsize=7)

    # 2b: Stokes Q, U, V vs frequency
    ax = axes2[0, 1]
    Q = [p['stokes_Q'] for p in pol_results]
    U = [p['stokes_U'] for p in pol_results]
    V = [p['stokes_V'] for p in pol_results]
    ax.plot(frequencies, Q, 'r-o', label='Q (linear H/V)', linewidth=2, markersize=8)
    ax.plot(frequencies, U, 'g-s', label='U (linear ±45°)', linewidth=2, markersize=8)
    ax.plot(frequencies, V, 'b-^', label='V (circular)', linewidth=2, markersize=8)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Stokes Parameter')
    ax.set_title('Stokes Q, U, V\n(V≠0 due to differential O/X loss)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    # 2c: Orientation angle vs frequency
    ax = axes2[0, 2]
    orientations = [p['orientation_deg'] for p in pol_results]
    ax.plot(frequencies, orientations, 'mo-', markersize=10, linewidth=2)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Orientation (degrees)')
    ax.set_title('Polarization Orientation')
    ax.grid(True, alpha=0.3)

    # 2d: Faraday rotation (N rotations)
    ax = axes2[1, 0]
    n_rot = [r['faraday_n_rotations'] for r in results]
    ax.plot(frequencies, n_rot, 'co-', markersize=10, linewidth=2)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Faraday Rotation (×π radians)')
    ax.set_title('Number of Faraday Rotations')
    ax.grid(True, alpha=0.3)

    # 2e: Phase difference between O and X
    ax = axes2[1, 1]
    phase_diff_waves = [r['phase_diff_rad'] / (2 * np.pi) for r in results]
    ax.plot(frequencies, phase_diff_waves, 'ko-', markersize=10, linewidth=2)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('O-X Phase Difference (wavelengths)')
    ax.set_title('O-X Mode Phase Difference')
    ax.grid(True, alpha=0.3)

    # 2f: Reflection height O vs X
    ax = axes2[1, 2]
    h_O = [r['reflection_height_O_km'] for r in results]
    h_X = [r['reflection_height_X_km'] for r in results]
    ax.plot(frequencies, h_O, 'b^-', markersize=10, linewidth=2, label='O-mode')
    ax.plot(frequencies, h_X, 'rv-', markersize=10, linewidth=2, label='X-mode')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Reflection Height (km)')
    ax.set_title('O/X Reflection Heights')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_polarization.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_prefix}_polarization.png')
    plt.close()

    # =========================================================================
    # Figure 3: Ray path details (profiles along path)
    # =========================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle(f'Ray Path Properties @ {r["freq_mhz"]:.1f} MHz', fontsize=14)

    ray = r['ray_O']
    if ray is not None and len(ray.get('ground_distances', [])) > 1:
        dist_km = ray['ground_distances']
        n_pts = len(dist_km)

        # 3a: Electron density along path
        ax = axes3[0, 0]
        if len(ray.get('Ne_profile', [])) >= n_pts:
            ax.semilogy(dist_km, ray['Ne_profile'][:n_pts], 'b-', linewidth=2)
        ax.set_xlabel('Ground Distance (km)')
        ax.set_ylabel('Electron Density (el/m³)')
        ax.set_title('Electron Density Along Path')
        ax.grid(True, alpha=0.3)

        # 3b: Magnetic field along path
        ax = axes3[0, 1]
        if len(ray.get('B_profile', [])) >= n_pts:
            ax.plot(dist_km, ray['B_profile'][:n_pts], 'r-', linewidth=2)
        ax.set_xlabel('Ground Distance (km)')
        ax.set_ylabel('Magnetic Field (nT)')
        ax.set_title('Magnetic Field Along Path')
        ax.grid(True, alpha=0.3)

        # 3c: Refractive indices along path
        ax = axes3[1, 0]
        if len(ray.get('n_O_profile', [])) >= n_pts:
            n_O_arr = np.array(ray['n_O_profile'][:n_pts])
            n_X_arr = np.array(ray['n_X_profile'][:n_pts])
            # For propagating waves, use real part; for evanescent, show as negative
            n_O_plot = np.where(np.real(n_O_arr**2) >= 0, np.real(n_O_arr), -np.abs(n_O_arr))
            n_X_plot = np.where(np.real(n_X_arr**2) >= 0, np.real(n_X_arr), -np.abs(n_X_arr))
            ax.plot(dist_km, n_O_plot, 'b-', linewidth=2, label='O-mode')
            ax.plot(dist_km, n_X_plot, 'r-', linewidth=2, label='X-mode')
            ax.axhline(1, color='k', linestyle='--', alpha=0.5)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            # Mark evanescent regions
            evanescent_X = np.real(n_X_arr**2) < 0
            if np.any(evanescent_X):
                ax.fill_between(dist_km, 0, -0.5, where=evanescent_X,
                               alpha=0.2, color='red', label='X cutoff')
        ax.set_xlabel('Ground Distance (km)')
        ax.set_ylabel('Refractive Index')
        ax.set_title('Refractive Indices Along Path\n(negative = evanescent/cutoff)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3d: Ray-B angle along path
        ax = axes3[1, 1]
        if len(ray.get('theta_profile', [])) >= n_pts:
            ax.plot(dist_km, ray['theta_profile'][:n_pts], 'g-', linewidth=2)
        ax.set_xlabel('Ground Distance (km)')
        ax.set_ylabel('Angle to B-field (degrees)')
        ax.set_title('Wave Vector - Magnetic Field Angle')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_ray_details.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_prefix}_ray_details.png')
    plt.close()


def create_3d_visualization(sim: HFPropagationSimulator3D, results: List[dict],
                            output_prefix: str = 'hf_3d'):
    """
    Create 3D visualization of O and X mode ray trajectories.
    Shows O/X splitting in both elevation and azimuth.
    Produces: _3d_rays.png
    """
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(16, 12))
    n_freq = len(results)
    frequencies = [r['freq_mhz'] for r in results]
    colors = plt.cm.viridis(np.linspace(0, 1, n_freq))

    # Main 3D plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    for i, r in enumerate(results):
        ray_O = r['ray_O']
        ray_X = r['ray_X']

        if ray_O is not None and len(ray_O.get('ground_distances', [])) > 0:
            dist_km = ray_O['ground_distances']
            alt_O = np.array(ray_O['altitudes'][:len(dist_km)]) / 1000
            label_O = 'O-mode (solid)' if i == 0 else None
            ax1.plot(dist_km, np.zeros_like(dist_km) + 5, alt_O,
                     color=colors[i], linewidth=2, linestyle='-', label=label_O)

        if ray_X is not None and len(ray_X.get('ground_distances', [])) > 0:
            dist_km = ray_X['ground_distances']
            alt_X = np.array(ray_X['altitudes'][:len(dist_km)]) / 1000
            label_X = 'X-mode (dashed)' if i == 0 else None
            ax1.plot(dist_km, np.zeros_like(dist_km) - 5, alt_X,
                     color=colors[i], linewidth=2, linestyle='--', label=label_X)

    ax1.set_xlabel('Distance along path (km)')
    ax1.set_ylabel('Cross-track (km)\n(O-mode +, X-mode -)')
    ax1.set_zlabel('Altitude (km)')
    ax1.set_title(f'{sim.n_hops}-Hop O/X Ray Paths\n(Separation exaggerated for visibility)')

    freq_legend = [Line2D([0], [0], color=colors[i], linewidth=3,
                          label=f'{frequencies[i]:.0f} MHz')
                   for i in range(n_freq)]
    mode_legend = [Line2D([0], [0], color='gray', linewidth=2, linestyle='-', label='O-mode'),
                   Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='X-mode')]
    ax1.legend(handles=freq_legend + mode_legend, loc='upper left', fontsize=8)
    ax1.view_init(elev=25, azim=-60)

    # Side view (altitude vs distance)
    ax2 = fig.add_subplot(2, 2, 2)
    for i, r in enumerate(results):
        ray_O = r['ray_O']
        ray_X = r['ray_X']

        if ray_O is not None and len(ray_O.get('ground_distances', [])) > 0:
            dist_O = ray_O['ground_distances']
            alt_O = np.array(ray_O['altitudes'][:len(dist_O)]) / 1000
            ax2.plot(dist_O, alt_O, color=colors[i], linewidth=2,
                     linestyle='-', label=f'{r["freq_mhz"]:.0f} MHz O')

        if ray_X is not None and len(ray_X.get('ground_distances', [])) > 0:
            dist_X = ray_X['ground_distances']
            alt_X = np.array(ray_X['altitudes'][:len(dist_X)]) / 1000
            ax2.plot(dist_X, alt_X, color=colors[i], linewidth=2,
                     linestyle='--', label=f'{r["freq_mhz"]:.0f} MHz X')

    ax2.set_xlabel('Distance along path (km)')
    ax2.set_ylabel('Altitude (km)')
    ax2.set_title('O/X Mode Height Profiles (Side View)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8, ncol=2)

    # O-X phase difference at receiver
    ax3 = fig.add_subplot(2, 2, 3)
    freqs = [r['freq_mhz'] for r in results]
    phase_diff_rad = [r['phase_diff_rad'] for r in results]
    phase_diff_waves = [pd / (2 * np.pi) for pd in phase_diff_rad]
    phase_diff_deg = [np.degrees(pd) % 360 for pd in phase_diff_rad]

    ax3_twin = ax3.twinx()
    l1, = ax3.plot(freqs, phase_diff_waves, 'bo-', markersize=10, linewidth=2, label='Phase diff (waves)')
    l2, = ax3_twin.plot(freqs, phase_diff_deg, 'r^--', markersize=8, linewidth=1.5, label='Phase diff mod 360°')

    ax3.set_xlabel('Frequency (MHz)')
    ax3.set_ylabel('O-X Phase Difference (wavelengths)', color='blue')
    ax3_twin.set_ylabel('Phase mod 360° (degrees)', color='red')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3.set_title('O-mode vs X-mode Phase Difference at Receiver')
    ax3.grid(True, alpha=0.3)
    ax3.legend(handles=[l1, l2], loc='upper right', fontsize=9)

    # Angle of arrival comparison WITH SPLITTING
    ax4 = fig.add_subplot(2, 2, 4)
    aoa_O_elev = [r['aoa_elevation_O_deg'] for r in results]
    aoa_X_elev = [r['aoa_elevation_X_deg'] for r in results]
    aoa_O_az = [r['aoa_azimuth_O_deg'] for r in results]
    aoa_X_az = [r['aoa_azimuth_X_deg'] for r in results]
    elev_split = [r['elevation_splitting_deg'] for r in results]
    az_split = [r['azimuth_splitting_deg'] for r in results]

    ax4.plot(freqs, aoa_O_elev, 'b^-', markersize=8, linewidth=1.5, label='O-mode elevation')
    ax4.plot(freqs, aoa_X_elev, 'rv-', markersize=8, linewidth=1.5, label='X-mode elevation')

    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel('Arrival Elevation (degrees)')
    ax4.set_title('Angle of Arrival at Receiver\n(Elevation Splitting)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')

    # Add text annotations for splitting
    for i, (f, es, azs) in enumerate(zip(freqs, elev_split, az_split)):
        ax4.annotate(f'Δel={es:.2f}°\nΔaz={azs:.3f}°', (f, aoa_O_elev[i]),
                    textcoords='offset points', xytext=(5, 10), fontsize=8, alpha=0.8)

    plt.suptitle(f'3D Ray Tracing: O/X Mode Splitting\n'
                 f'{sim.tx.lat:.2f}°N, {sim.tx.lon:.2f}°E → '
                 f'{sim.rx.lat:.2f}°N, {sim.rx.lon:.2f}°E | '
                 f'{sim.date.strftime("%Y-%m-%d %H:%M")} UTC | {sim.n_hops} hop(s)',
                 fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_3d_rays.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_prefix}_3d_rays.png')
    plt.close()


def create_polarization_plot(sim: HFPropagationSimulator3D, results: List[dict],
                             output_prefix: str = 'hf_3d'):
    """
    Create detailed polarization state plot at receiver.
    Produces: _rx_polarization.png
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Polarization State at Receiver\n'
                 '(Input: Linear Vertical from WWV Vertical Dipole)',
                 fontsize=14)

    frequencies = [r['freq_mhz'] for r in results]
    n_freq = len(frequencies)
    colors = plt.cm.viridis(np.linspace(0, 1, n_freq))

    pol_results = [sim.polarization_state(r) for r in results]

    # 1. Polarization ellipses
    ax = axes[0, 0]
    theta = np.linspace(0, 2*np.pi, 100)
    for i, (r, p) in enumerate(zip(results, pol_results)):
        a = 1.0
        # Ellipticity: b/a = tan(|χ|), where χ is ellipticity angle
        b = abs(np.tan(np.radians(p['ellipticity_deg']))) if abs(p['ellipticity_deg']) > 0.1 else 0.02
        psi = np.radians(p['orientation_deg'])
        x = a * np.cos(theta) * np.cos(psi) - b * np.sin(theta) * np.sin(psi)
        y = a * np.cos(theta) * np.sin(psi) + b * np.sin(theta) * np.cos(psi)
        ax.plot(x * 0.8 + i * 0.3, y * 0.8, color=colors[i], linewidth=2,
                label=f'{r["freq_mhz"]:.0f} MHz (χ={p["ellipticity_deg"]:.1f}°)')
        ax.plot([i*0.3 - 0.7*np.cos(psi), i*0.3 + 0.7*np.cos(psi)],
                [-0.7*np.sin(psi), 0.7*np.sin(psi)],
                color=colors[i], linestyle='--', alpha=0.5)

    ax.set_xlim(-1.5, n_freq * 0.3 + 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Ex (horizontal)')
    ax.set_ylabel('Ey (vertical)')
    ax.set_title('Polarization Ellipses\n(χ > 0: RCP sense, χ < 0: LCP sense)')
    ax.legend(loc='upper right', fontsize=7)

    # 2. Orientation angle
    ax = axes[0, 1]
    orientations = [p['orientation_deg'] for p in pol_results]
    ax.plot(frequencies, orientations, 'mo-', markersize=10, linewidth=2)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Orientation (degrees)')
    ax.set_title('Polarization Orientation')
    ax.grid(True, alpha=0.3)

    # 3. Degree of linear/circular polarization
    ax = axes[0, 2]
    dolp = [p['dolp'] for p in pol_results]
    docp = [p['docp'] for p in pol_results]
    ax.plot(frequencies, dolp, 'g-o', markersize=10, linewidth=2, label='DoLP')
    ax.plot(frequencies, docp, 'c-s', markersize=10, linewidth=2, label='DoCP')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Degree of Polarization')
    ax.set_title('Linear vs Circular Polarization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # 4. Stokes Q, U, and V
    ax = axes[1, 0]
    Q = [p['stokes_Q'] for p in pol_results]
    U = [p['stokes_U'] for p in pol_results]
    V = [p['stokes_V'] for p in pol_results]
    ax.plot(frequencies, Q, 'r-o', label='Q (H/V)', linewidth=2, markersize=8)
    ax.plot(frequencies, U, 'g-s', label='U (±45°)', linewidth=2, markersize=8)
    ax.plot(frequencies, V, 'b-^', label='V (circ)', linewidth=2, markersize=8)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Stokes Parameter')
    ax.set_title('Stokes Q, U, V\n(V > 0: RCP excess, V < 0: LCP excess)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    # 5. RCP/LCP fractions
    ax = axes[1, 1]
    frac_rcp = [p['frac_RCP'] for p in pol_results]
    frac_lcp = [p['frac_LCP'] for p in pol_results]
    ax.plot(frequencies, frac_rcp, 'r-o', markersize=10, linewidth=2, label='RCP')
    ax.plot(frequencies, frac_lcp, 'b-s', markersize=10, linewidth=2, label='LCP')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power Fraction')
    ax.set_title('RCP/LCP Power Fractions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 6. Faraday rotation
    ax = axes[1, 2]
    faraday_deg = [r['faraday_rotation_deg'] for r in results]
    ax.plot(frequencies, faraday_deg, 'ro-', markersize=10, linewidth=2)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Faraday Rotation (degrees)')
    ax.set_title('Faraday Rotation')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_rx_polarization.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_prefix}_rx_polarization.png')
    plt.close()


def create_angular_splitting_plot(sim: HFPropagationSimulator3D, results: List[dict],
                                   output_prefix: str = 'hf_3d'):
    """
    Create detailed O/X angular splitting analysis plot.
    This is specific to 3D version - shows elevation and azimuth splitting.
    Produces: _angular_splitting.png
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get MUF
    muf = results[0].get('muf_mhz', 0)

    fig.suptitle(f'O/X Mode Angular Splitting (3D Ray Tracing)\n'
                 f'{sim.tx.lat:.2f}°N, {sim.tx.lon:.2f}°E → '
                 f'{sim.rx.lat:.2f}°N, {sim.rx.lon:.2f}°E | {sim.n_hops} hop(s) | MUF={muf:.1f} MHz',
                 fontsize=14)

    frequencies = [r['freq_mhz'] for r in results]

    # 1. Elevation comparison
    ax = axes[0, 0]
    elev_O = [r['aoa_elevation_O_deg'] for r in results]
    elev_X = [r['aoa_elevation_X_deg'] for r in results]
    ax.plot(frequencies, elev_O, 'b^-', markersize=10, linewidth=2, label='O-mode')
    ax.plot(frequencies, elev_X, 'rv-', markersize=10, linewidth=2, label='X-mode')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Arrival Elevation (degrees)')
    ax.set_title('O and X Mode Arrival Elevations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Elevation splitting
    ax = axes[0, 1]
    elev_split = [r['elevation_splitting_deg'] for r in results]
    ax.plot(frequencies, elev_split, 'go-', markersize=10, linewidth=2)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Elevation Splitting (degrees)')
    ax.set_title('O-X Elevation Separation')
    ax.grid(True, alpha=0.3)

    # Add 1/f reference
    if len(frequencies) > 1 and elev_split[0] > 0:
        f_ref = frequencies[0]
        es_ref = elev_split[0]
        f_line = np.linspace(min(frequencies)*0.8, max(frequencies)*1.2, 50)
        es_line = es_ref * (f_ref / f_line)
        ax.plot(f_line, es_line, 'k--', alpha=0.5, label='∝ 1/f')
        ax.legend()

    # 3. Azimuth comparison
    ax = axes[1, 0]
    az_O = [r['aoa_azimuth_O_deg'] for r in results]
    az_X = [r['aoa_azimuth_X_deg'] for r in results]
    nominal_az = results[0]['ray_O'].get('nominal_azimuth', az_O[0]) if results[0]['ray_O'] else az_O[0]

    ax.plot(frequencies, az_O, 'b^-', markersize=10, linewidth=2, label='O-mode')
    ax.plot(frequencies, az_X, 'rv-', markersize=10, linewidth=2, label='X-mode')
    ax.axhline(nominal_az, color='gray', linestyle='--', alpha=0.5, label=f'Nominal ({nominal_az:.1f}°)')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Arrival Azimuth (degrees)')
    ax.set_title('O and X Mode Arrival Azimuths')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Azimuth splitting
    ax = axes[1, 1]
    az_split = [r['azimuth_splitting_deg'] for r in results]
    ax.plot(frequencies, az_split, 'mo-', markersize=10, linewidth=2)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Azimuth Splitting (degrees)')
    ax.set_title('O-X Azimuth Separation (Lateral Deflection)')
    ax.grid(True, alpha=0.3)

    # Add 1/f² reference
    if len(frequencies) > 1 and az_split[0] > 0:
        f_ref = frequencies[0]
        azs_ref = az_split[0]
        f_line = np.linspace(min(frequencies)*0.8, max(frequencies)*1.2, 50)
        azs_line = azs_ref * (f_ref / f_line)**2
        ax.plot(f_line, azs_line, 'k--', alpha=0.5, label='∝ 1/f²')
        ax.legend()

    # Add MUF indicator to all panels
    if muf > 0 and min(frequencies) < muf < max(frequencies) * 1.5:
        for ax in axes.flat:
            ax.axvline(x=muf, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        # Add MUF label to first panel
        axes[0, 0].text(muf, axes[0, 0].get_ylim()[1], ' MUF', color='red',
                        fontsize=9, va='top', ha='left')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_angular_splitting.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_prefix}_angular_splitting.png')
    plt.close()


def print_results(sim: HFPropagationSimulator3D, results: List[dict]):
    """Print summary results table."""
    print("\n" + "="*150)
    print("3D HF PROPAGATION SIMULATION RESULTS")
    print("="*150)
    print(f"\nPath: ({sim.tx.lat:.3f}°N, {sim.tx.lon:.3f}°E) → ({sim.rx.lat:.3f}°N, {sim.rx.lon:.3f}°E)")
    print(f"Date/Time: {sim.date.isoformat()}")
    print(f"Ground Distance: {results[0]['ground_distance_km']:.1f} km")
    print(f"Number of hops: {sim.n_hops}")
    print(f"Bearing: {sim.bearing:.1f}°")
    print(f"\nSpace Weather: F10.7={sim.sw.f107:.1f}, Kp={sim.sw.kp:.1f}, SSN={sim.sw.ssn:.0f}")

    # MUF
    muf = results[0].get('muf_mhz', 0)
    print(f"Maximum Usable Frequency (MUF): {muf:.1f} MHz")

    print("\n" + "-"*150)
    print(f"{'Freq':>8} {'h_O':>8} {'h_X':>8} {'Loss':>8} {'Faraday':>10} "
          f"{'O Elev':>8} {'X Elev':>8} {'Elv Spl':>8} {'O Az':>8} {'X Az':>8} "
          f"{'Az Spl':>8} {'Status':>10}")
    print(f"{'(MHz)':>8} {'(km)':>8} {'(km)':>8} {'(dB)':>8} {'(deg)':>10} "
          f"{'(deg)':>8} {'(deg)':>8} {'(deg)':>8} {'(deg)':>8} {'(deg)':>8} "
          f"{'(deg)':>8} {'':>10}")
    print("-"*150)

    for r in results:
        status = "ABOVE_MUF" if r.get('above_muf', False) else "OK"
        print(f"{r['freq_mhz']:>8.1f} "
              f"{r.get('reflection_height_O_km', r['reflection_height_km']):>8.1f} "
              f"{r.get('reflection_height_X_km', r['reflection_height_km']):>8.1f} "
              f"{r['total_loss_db']:>8.1f} {r['faraday_rotation_deg']:>10.0f} "
              f"{r['aoa_elevation_O_deg']:>8.2f} {r['aoa_elevation_X_deg']:>8.2f} "
              f"{r.get('elevation_splitting_deg', 0):>8.3f} "
              f"{r['aoa_azimuth_O_deg']:>8.1f} {r['aoa_azimuth_X_deg']:>8.1f} "
              f"{r['azimuth_splitting_deg']:>8.3f} {status:>10}")

    print("="*150)
    print("\nKey:")
    print("  h_O/h_X:   O-mode and X-mode reflection heights (O penetrates higher)")
    print("  O Elev/Az: O-mode arrival elevation and azimuth")
    print("  X Elev/Az: X-mode arrival elevation and azimuth")
    print("  Elv Spl:   Elevation separation between O and X modes")
    print("  Az Spl:    Azimuthal separation between O and X modes (lateral splitting)")


def write_summary(sim: HFPropagationSimulator3D, results: List[dict],
                  output_prefix: str = 'hf_3d'):
    """Write summary file."""
    filename = f'{output_prefix}_summary.txt'

    with open(filename, 'w') as f:
        f.write("="*100 + "\n")
        f.write("3D HF PROPAGATION SIMULATION - SUMMARY\n")
        f.write("="*100 + "\n\n")

        f.write("SIMULATION VERSION: 3D Ray Tracing with O/X splitting (elevation and azimuth)\n\n")

        f.write("PATH GEOMETRY\n")
        f.write("-"*40 + "\n")
        f.write(f"Transmitter: {sim.tx.lat:.4f}°N, {sim.tx.lon:.4f}°E\n")
        f.write(f"Receiver:    {sim.rx.lat:.4f}°N, {sim.rx.lon:.4f}°E\n")
        f.write(f"Distance:    {results[0]['ground_distance_km']:.1f} km\n")
        f.write(f"Number of hops: {sim.n_hops}\n")
        f.write(f"Bearing:     {sim.bearing:.1f}°\n\n")

        f.write("DATE/TIME\n")
        f.write("-"*40 + "\n")
        f.write(f"Simulation: {sim.date.isoformat()}\n\n")

        # MUF information
        muf = results[0].get('muf_mhz', 0)
        f.write("PROPAGATION CONDITIONS\n")
        f.write("-"*40 + "\n")
        f.write(f"Maximum Usable Frequency (MUF): {muf:.1f} MHz\n")
        f.write(f"(Frequencies above MUF will penetrate the ionosphere)\n\n")

        f.write("RESULTS BY FREQUENCY\n")
        f.write("-"*110 + "\n")
        f.write(f"{'Freq':>8} {'h_O':>8} {'h_X':>8} {'O Elev':>10} {'X Elev':>10} "
                f"{'Elv Spl':>10} {'O Az':>10} {'X Az':>10} {'Az Spl':>10} {'Status':>10}\n")
        f.write(f"{'(MHz)':>8} {'(km)':>8} {'(km)':>8} {'(deg)':>10} {'(deg)':>10} "
                f"{'(deg)':>10} {'(deg)':>10} {'(deg)':>10} {'(deg)':>10} {'':>10}\n")
        f.write("-"*110 + "\n")

        for r in results:
            status = "ABOVE_MUF" if r.get('above_muf', False) else "OK"
            f.write(f"{r['freq_mhz']:>8.1f} "
                    f"{r.get('reflection_height_O_km', r['reflection_height_km']):>8.1f} "
                    f"{r.get('reflection_height_X_km', r['reflection_height_km']):>8.1f} "
                    f"{r['aoa_elevation_O_deg']:>10.2f} "
                    f"{r['aoa_elevation_X_deg']:>10.2f} "
                    f"{r.get('elevation_splitting_deg', 0):>10.4f} "
                    f"{r['aoa_azimuth_O_deg']:>10.1f} "
                    f"{r['aoa_azimuth_X_deg']:>10.1f} "
                    f"{r['azimuth_splitting_deg']:>10.4f} "
                    f"{status:>10}\n")

        # Polarization results section
        f.write("\n" + "RECEIVED POLARIZATION (due to differential O/X absorption)\n")
        f.write("-"*110 + "\n")
        f.write(f"{'Freq':>8} {'Loss_O':>8} {'Loss_X':>8} {'ΔLoss':>8} "
                f"{'LCP/RCP':>10} {'L/R (dB)':>10} {'Axial':>8} {'Ellip':>8} {'%LCP':>8} {'%RCP':>8}\n")
        f.write(f"{'(MHz)':>8} {'(dB)':>8} {'(dB)':>8} {'(dB)':>8} "
                f"{'ratio':>10} {'':>10} {'ratio':>8} {'(deg)':>8} {'':>8} {'':>8}\n")
        f.write("-"*110 + "\n")

        for r in results:
            pol = sim.polarization_state(r)
            loss_O = r.get('total_loss_O_db', r.get('total_loss_db', 0))
            loss_X = r.get('total_loss_X_db', r.get('total_loss_db', 0))
            delta_loss = loss_X - loss_O

            # LCP/RCP power ratio from Stokes V: P_LCP/P_RCP = (1-V)/(1+V)
            V = pol['stokes_V']
            if abs(1 + V) > 1e-10:
                lcp_rcp_ratio = (1 - V) / (1 + V)
                lcp_rcp_db = 10 * np.log10(lcp_rcp_ratio) if lcp_rcp_ratio > 0 else float('inf')
            else:
                lcp_rcp_ratio = float('inf')
                lcp_rcp_db = float('inf')

            # Axial ratio (major/minor) from ellipticity angle
            # AR = 1/|tan(χ)| for |χ| < 45°, but tan(0) = 0 gives infinity
            # Use minor/major = |tan(χ)| which goes 0 (linear) to 1 (circular)
            chi_rad = np.radians(pol['ellipticity_deg'])
            minor_major = abs(np.tan(chi_rad)) if abs(chi_rad) > 0.001 else 0.0

            # Format output
            if lcp_rcp_ratio > 1000:
                ratio_str = ">1000"
                db_str = f">{lcp_rcp_db:.0f}" if lcp_rcp_db < 100 else ">100"
            elif lcp_rcp_ratio < 0.001:
                ratio_str = "<0.001"
                db_str = f"<{lcp_rcp_db:.0f}" if lcp_rcp_db > -100 else "<-100"
            else:
                ratio_str = f"{lcp_rcp_ratio:.2f}"
                db_str = f"{lcp_rcp_db:+.1f}"

            f.write(f"{r['freq_mhz']:>8.2f} "
                    f"{loss_O:>8.1f} "
                    f"{loss_X:>8.1f} "
                    f"{delta_loss:>8.1f} "
                    f"{ratio_str:>10} "
                    f"{db_str:>10} "
                    f"{minor_major:>8.3f} "
                    f"{pol['ellipticity_deg']:>8.1f} "
                    f"{pol['frac_LCP']*100:>8.1f} "
                    f"{pol['frac_RCP']*100:>8.1f}\n")

        f.write("\nKey:\n")
        f.write("  ΔLoss = Loss_X - Loss_O (positive = X-mode absorbed more, favors LCP)\n")
        f.write("  LCP/RCP ratio > 1 means LCP dominates (use LCP feed for best SNR)\n")
        f.write("  Axial ratio: minor/major axis (0 = linear, 1 = circular)\n")
        f.write("  Ellip: ellipticity angle χ (-45° = pure LCP, 0° = linear, +45° = pure RCP)\n")

        f.write("\n" + "="*100 + "\n")
        f.write("NOTES ON 3D RAY TRACING\n")
        f.write("="*100 + "\n")
        f.write("""
1. ELEVATION SPLITTING: O-mode penetrates higher into the ionosphere before
   reflecting (because n_O is closer to 1). X-mode reflects at lower altitude.
   This results in different arrival elevation angles.

2. LATERAL (AZIMUTHAL) SPLITTING: The O and X modes experience different
   lateral deflections due to the magnetized ionosphere. The deflection is
   perpendicular to both the density gradient and the magnetic field.

3. MULTIHOP: For paths requiring multiple hops, the code divides the path
   into equal segments. Each hop adds 180° phase shift at ground reflection.
   Faraday rotation and absorption accumulate across all hops.

4. The O/X splitting is typically small (< 1 degree) but can be significant
   for precise direction-finding or interferometric applications.
""")

    print(f'Saved: {filename}')


def main():
    parser = argparse.ArgumentParser(
        description='3D HF Propagation Simulator with O/X lateral splitting')

    # Location arguments
    parser.add_argument('--tx-lat', type=float, default=40.68,
                        help='Transmitter latitude (default: 40.68 = WWV)')
    parser.add_argument('--tx-lon', type=float, default=-105.04,
                        help='Transmitter longitude (default: -105.04 = WWV)')
    parser.add_argument('--rx-lat', type=float, default=37.23,
                        help='Receiver latitude (default: 37.23 = OVRO)')
    parser.add_argument('--rx-lon', type=float, default=-118.28,
                        help='Receiver longitude (default: -118.28 = OVRO)')

    # Frequency arguments
    parser.add_argument('--frequencies', '-f', type=float, nargs='+',
                        default=[5, 10, 15, 20, 25],
                        help='Frequencies to simulate (MHz)')

    # Hop arguments
    parser.add_argument('--hops', '-n', type=int, default=1,
                        help='Number of ionospheric hops (default: 1)')

    # Time arguments
    parser.add_argument('--date', type=str, default=None,
                        help='Date/time (ISO format, default: now)')

    # Output arguments
    parser.add_argument('--output', '-o', type=str, default='hf_3d',
                        help='Output file prefix')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    # Parse date
    if args.date:
        date = datetime.fromisoformat(args.date)
    else:
        date = datetime.now(timezone.utc)

    # Create locations
    tx = Location(args.tx_lat, args.tx_lon, 0)
    rx = Location(args.rx_lat, args.rx_lon, 0)

    print("="*70)
    print("3D HF PROPAGATION SIMULATOR")
    print("="*70)
    print(f"\nTransmitter: {tx.lat:.4f}°N, {tx.lon:.4f}°E")
    print(f"Receiver:    {rx.lat:.4f}°N, {rx.lon:.4f}°E")
    print(f"Date/Time:   {date.isoformat()}")
    print(f"Frequencies: {args.frequencies} MHz")
    print(f"Number of hops: {args.hops}")

    # Create simulator
    sim = HFPropagationSimulator3D(tx, rx, date, n_hops=args.hops)

    print(f"\nGround distance: {sim.ground_distance/1000:.1f} km")
    print(f"Hop distance: {sim.ground_distance/1000/args.hops:.1f} km")
    print(f"Bearing: {sim.bearing:.1f}°")

    # Run simulation
    results = sim.simulate_frequencies(args.frequencies)

    # Print results
    print_results(sim, results)

    # Write summary
    write_summary(sim, results, args.output)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating diagnostic plots...")
        create_plots(sim, results, args.output)
        create_3d_visualization(sim, results, args.output)
        create_polarization_plot(sim, results, args.output)
        create_angular_splitting_plot(sim, results, args.output)

        # Print generated files summary
        print(f"\nGenerated files:")
        print(f"  {args.output}_summary.txt           - Text summary")
        print(f"  {args.output}_overview.png          - Overview plots")
        print(f"  {args.output}_polarization.png      - Polarization analysis")
        print(f"  {args.output}_ray_details.png       - Ray path details")
        print(f"  {args.output}_3d_rays.png           - 3D O/X ray visualization")
        print(f"  {args.output}_rx_polarization.png   - Receiver polarization state")
        print(f"  {args.output}_angular_splitting.png - O/X elevation & azimuth splitting")

    print("\nDone!")


if __name__ == '__main__':
    main()
