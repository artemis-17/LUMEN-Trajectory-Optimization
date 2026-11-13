
import numpy as np
import plotly.graph_objects as go
import spiceypy as spice

from solar_sail import AU

# Planet name, SPICE target, orbital period (days), color
PLANETS = [
    ("Mercury", "MERCURY BARYCENTER", 87.97, "#A0A0A0"),
    ("Venus", "VENUS BARYCENTER", 224.70, "#E5BD79"),
    ("Earth", "EARTH", 365.26, "#4285F4"),
    ("Mars", "MARS BARYCENTER", 686.98, "#DE7755"),
    ("Jupiter", "JUPITER BARYCENTER", 4332.59, "#D1A984"),
    ("Saturn", "SATURN BARYCENTER", 10759.22, "#DCC894"),
    ("Uranus", "URANUS BARYCENTER", 30688.5, "#77D5DA"),
    ("Neptune", "NEPTUNE BARYCENTER", 60182.0, "#648CF0")
]

def plot_trajectory(r_sc, et0):

    fig = go.Figure()

    # Rotate spacecraft positions from J2000 to ECLIPJ2000
    R = spice.pxform("J2000", "ECLIPJ2000", et0)
    r_sc = (np.asarray(r_sc) @ R.T) / AU

    # Planetary orbits and starting positions
    for name, target, period, color in PLANETS:

        # Approximate one full orbital period centered on et0
        times = et0 + np.linspace(-period/2, period/2, 300) * 86400.0

        orbit = np.array([
            spice.spkpos(target, float(t), "ECLIPJ2000",
                         "NONE", "SUN")[0]
            for t in times
        ]) * 1000.0 / AU

        pos = np.array(
            spice.spkpos(target, et0, "ECLIPJ2000",
                         "NONE", "SUN")[0]
        ) * 1000.0 / AU

        fig.add_trace(go.Scatter3d(
            x=orbit[:, 0], y=orbit[:, 1], z=orbit[:, 2],
            mode="lines",
            name=f"{name} Orbit",
            line=dict(color=color, width=3),
            showlegend=False,
            legendgroup=name
        ))

        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers",
            name=name,
            marker=dict(size=5, color=color),
            legendgroup=name
        ))

    # Spacecraft trajectory
    fig.add_trace(go.Scatter3d(
        x=r_sc[:, 0], y=r_sc[:, 1], z=r_sc[:, 2],
        mode="lines",
        name="LUMEN",
        line=dict(color="cyan", width=6)
    ))

    fig.add_trace(go.Scatter3d(
        x=[r_sc[0, 0]], y=[r_sc[0, 1]], z=[r_sc[0, 2]],
        mode="markers",
        name="Start",
        marker=dict(size=6, color="lime")
    ))

    fig.add_trace(go.Scatter3d(
        x=[r_sc[-1, 0]], y=[r_sc[-1, 1]], z=[r_sc[-1, 2]],
        mode="markers",
        name="End",
        marker=dict(size=6, color="red")
    ))

    # Sun
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        name="Sun",
        marker=dict(size=11, color="orange")
    ))

    fig.update_layout(
        title="LUMEN | Solar System Trajectory",
        template="plotly_dark",
        scene=dict(
            xaxis_title="X [AU]",
            yaxis_title="Y [AU]",
            zaxis_title="Z [AU]",
            aspectmode="data",
            bgcolor="rgb(8, 12, 24)"
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show(renderer="browser")
