import pandas as pd
import plotly.graph_objects as go


def plot_SFR_reach_TS(sub_title, X_axis, d_plot, Pa_Out):
    fig = go.Figure()

    y_vals = []
    y_2_vals = []
    for k, v in d_plot.items():
        # Prepare kwargs
        kwargs = v['kwargs'].copy()
        kwargs['name'] = k
        y_data = v['y']

        # Collect data for auto-scaling
        if 'yaxis' in kwargs and kwargs['yaxis'] == 'y2':
            y_2_vals.extend(y_data)
        else:
            # Handle potential None/NaNs in data before extending default list
            if hasattr(y_data, 'dropna'):
                y_vals.extend(y_data.dropna().values)
            else:
                y_vals.extend([y for y in y_data if pd.notna(y)])

        # Plot
        fig.add_trace(v['plot_type'](x=X_axis, y=y_data, **kwargs))

    # ---------------------------
    # Auto-Scaling Calculation for Bottom Plot
    if y_vals:
        y_min, y_max = min(y_vals), max(y_vals)
        # Add 5% padding
        y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.5
        y_vals_range = [y_min - y_pad, y_max + y_pad]
    else:
        y_vals_range = [0, 1]

    fig.update_layout(
        title=dict(
            text=f'SFR Vs RIV stage<br><span style="font-size: 14px; color: gray;">{sub_title}</span>',
            x=0.5,
            xanchor='center',
            y=0.98,  # Position title just above the plot area/below legend
            yanchor='top',
        ),
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02,
            font=dict(family='Consolas, monospace', size=12),  # Monospace for alignment
        ),
        hovermode='x unified',  # Unified hover for single window
        # Ensure hover searches across the whole column regardless of cursor Y position
        hoverdistance=-1,
        spikedistance=-1,
        hoverlabel=dict(
            namelength=-1, bgcolor='white', bordercolor='gray', font=dict(family='Consolas, monospace'), align='right'
        ),
        template='plotly_white',
        margin=dict(t=80, l=60, r=40, b=40),  # Decreased top margin slightly as requested (180->160)
        # X Axis (Shared)
        xaxis=dict(
            domain=[0, 1],
            anchor='free',  # Anchor to free to allow unified hover across disjoint y-domains
            position=0,
            dtick='M1',  # Ticks every month
            tickformat='%b %Y',  # e.g., 'Jan 1994'
            tickangle=-90,  # Vertical text
            hoverformat='%d %b %Y',  # Full date in hover
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='gray',
            spikethickness=1,
            showline=True,
            linewidth=1,
            linecolor='gray',
            mirror=False,
            showgrid=True,
        ),
        # Y Axis (Elevation - Bottom)
        yaxis=dict(
            title_text='Elevation (mNAP)',
            domain=[0, 0.68],
            range=y_vals_range,
            nticks=15,  # Force more ticks/gridlines
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='gray',
            spikethickness=1,
            showline=True,
            linewidth=1,
            linecolor='gray',
            mirror=False,
            showgrid=True,
            fixedrange=False,  # Ensure interactive
        ),
        # Y Axis 2 (Precipitation - Top)
        yaxis2=dict(
            title_text='Precipitation (mm)',
            domain=[0.72, 1],
            # max=None, min=0 auto
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='gray',
            spikethickness=1,
            showline=True,
            linewidth=1,
            linecolor='gray',
            mirror=False,
            showgrid=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            fixedrange=False,  # Ensure interactive
        ),
    )

    fig.write_html(Pa_Out, auto_open=True)
