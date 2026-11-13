"""Standalone synchronized trajectory and sail geometry animation."""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import spiceypy as spice
from mission import DAY
from ephemerides import NAMES, body_state
from solar_sail import AU


def animate_rollout(result, et0, output='animation.html', max_frames=400):
    t, X, commands = result['t'], result['X'].T, result['commands']
    indices = np.unique(np.linspace(0, len(t)-1, min(max_frames, len(t))).astype(int))
    R = spice.pxform('J2000', 'ECLIPJ2000', et0)
    track = X[:, :3] @ R.T / AU
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'scene'}, {'type':'scene'}]],
                        subplot_titles=('Heliocentric trajectory (AU)', 'Sail and Sun direction (unit vectors)'))
    fig.add_trace(go.Scatter3d(x=track[:,0], y=track[:,1], z=track[:,2],
                             mode='lines', name='Trajectory', line=dict(color='cyan')), 1, 1)
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', name='Sun',
                             marker=dict(color='orange', size=9)), 1, 1)
    # Dynamic traces: spacecraft, planets, sail mesh, normal, spacecraft-to-Sun vector.
    def traces(i):
        r, v = X[i,:3], X[i,3:]
        er = r/np.linalg.norm(r)
        eh = np.cross(r,v); eh /= np.linalg.norm(eh)
        et = np.cross(eh,er)
        j = max(0, np.searchsorted(commands[:,0], t[i], side='right')-1)
        a, d = commands[j,1:]
        normal = np.cos(a)*er + np.sin(a)*(np.cos(d)*et + np.sin(d)*eh)
        u = np.cross(normal, eh)
        if np.linalg.norm(u) < 1e-10:
            u = np.cross(normal, et)
        u /= np.linalg.norm(u)
        w = np.cross(normal, u)
        corners = np.array([-u-w, u-w, u+w, -u+w])*0.5 @ R.T
        normal = normal @ R.T
        sun = -er @ R.T
        planets = np.array([body_state(name, et0+t[i])[:3] for name in NAMES]) @ R.T/AU
        def vector(vec, name, color):
            return go.Scatter3d(x=[0,vec[0]], y=[0,vec[1]], z=[0,vec[2]],
                                mode='lines+markers', name=name, line=dict(color=color,width=7))
        return [go.Scatter3d(x=[track[i,0]], y=[track[i,1]], z=[track[i,2]],
                             mode='markers', name='Spacecraft', marker=dict(color='lime',size=5)),
                go.Scatter3d(x=planets[:,0], y=planets[:,1], z=planets[:,2],
                             mode='markers+text', text=list(NAMES), name='Planets', marker=dict(size=4)),
                go.Mesh3d(x=corners[:,0], y=corners[:,1], z=corners[:,2],
                          i=[0,0], j=[1,2], k=[2,3], name='Sail', color='silver', opacity=0.75),
                vector(normal, 'Sail normal', 'cyan'), vector(sun, 'Toward Sun', 'orange')]
    initial = traces(0)
    for k, trace in enumerate(initial):
        fig.add_trace(trace, 1, 1 if k < 2 else 2)
    frames = []
    for i in indices:
        j = max(0, np.searchsorted(commands[:,0], t[i], side='right')-1)
        a,d = np.rad2deg(commands[j,1:])
        frames.append(go.Frame(name=str(i), data=traces(i), traces=[2,3,4,5,6],
                               layout=go.Layout(title_text=f'Day {t[i]/DAY:.1f} | cone {a:.1f}° | clock {d:.1f}°')))
    fig.frames = frames
    fig.update_layout(template='plotly_dark', title='Solar sail mission',
                      scene=dict(aspectmode='data', xaxis_title='X [AU]', yaxis_title='Y [AU]', zaxis_title='Z [AU]'),
                      scene2=dict(aspectmode='cube', xaxis=dict(range=[-1.2,1.2]),
                                  yaxis=dict(range=[-1.2,1.2]), zaxis=dict(range=[-1.2,1.2])),
                      uirevision='keep-camera',
                      updatemenus=[dict(type='buttons', buttons=[
                          dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=80,redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                          dict(label='Pause', method='animate', args=[[None], dict(mode='immediate',frame=dict(duration=0,redraw=False))])])],
                      sliders=[dict(steps=[dict(label=f'{t[i]/DAY:.0f} d', method='animate',
                                               args=[[str(i)],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for i in indices])])
    fig.write_html(str(Path(output)), include_plotlyjs=True, auto_play=False)
    print(f'Animation saved: {Path(output).resolve()}')
    return fig
