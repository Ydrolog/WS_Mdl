import re
from datetime import datetime as DT
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from WS_Mdl.core.style import sprint


def SFR_reach_TS(sub_title, X_axis, d_plot, Pa_Out):
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


def range(l_Pa, ending='IDF', date_format='%Y%m%d', Out_Fi='TS_range.png'):
    """Reads file names using a naming convention containing dates from multiple directories.
    Then plots the time series range as an image with one line per directory.
    Uses regular expressions to extract dates from filenames, making it more versatile than assuming
    the date is always the second element when splitting by underscore.

    Parameters:
    -----------
    l_Pa : str or list
        Single path (str) or list of paths to directories containing files with dates
    ending : str
        File extension to filter by (default: 'IDF')
    date_format : str
        Date format pattern for parsing dates from filenames (default: '%Y%m%d')
    Out_Fi : str
        Output filename for the plot (default: 'TS_range.png')
    """

    # Handle single path input by converting to list
    if isinstance(l_Pa, str):
        l_Pa = [l_Pa]
    l_Pa = [Path(p) for p in l_Pa]  # Convert to Path objects

    # Create regex pattern based on date_format
    date_pattern = date_format.replace('%Y', r'\d{4}').replace('%m', r'\d{2}').replace('%d', r'\d{2}')

    # Collect data for each path
    data_by_path = {}
    all_dates = []

    for Pa in l_Pa:
        if not Pa.exists():
            sprint(f'Warning: Path does not exist: {Pa}')
            continue

        l_Fi = [f for f in Pa.iterdir() if f.is_file() and f.name.endswith(ending)]
        l_Dt = []

        for f in l_Fi:
            # Search for date pattern in filename
            match = re.search(date_pattern, f.name)
            if match:
                try:
                    date_str = match.group(0)
                    dt = DT.strptime(date_str, date_format)
                    l_Dt.append(dt)
                    all_dates.append(dt)
                except ValueError:
                    sprint(f'Warning: Could not parse date from filename: {f}')
            else:
                sprint(f'Warning: No date pattern found in filename: {f}')

        if l_Dt:
            l_Dt.sort()
            data_by_path[Pa] = l_Dt
            sprint(
                f'Found {len(l_Dt)} files with dates in {Pa.name} ranging from {l_Dt[0].date()} to {l_Dt[-1].date()}'
            )
        else:
            sprint(f'No valid dates found in {Pa}')

    if not data_by_path:
        print('No valid dates found in any of the provided paths')
        return

    # Create the plot

    fig = go.Figure()

    for Pa, l_Dt in data_by_path.items():
        # Prepare data with gaps
        x_vals = []
        y_vals = []

        if l_Dt:
            x_vals.append(l_Dt[0])
            y_vals.append(Pa.name)

            for j in range(len(l_Dt) - 1):
                current_date = l_Dt[j]
                next_date = l_Dt[j + 1]
                days_diff = (next_date - current_date).days

                if days_diff > 7:
                    # Insert None to break line
                    x_vals.append(None)
                    y_vals.append(None)

                x_vals.append(next_date)
                y_vals.append(Pa.name)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name=f'{Pa.name} ({len(l_Dt)} files)',
                marker=dict(size=5),
                line=dict(width=2),
                hovertemplate='%{x|%Y-%m-%d}<br>%{y}',
            )
        )

    fig.update_layout(
        title='Time Series Range Comparison',
        xaxis_title='Date',
        yaxis_title='Directory',
        hovermode='closest',
        template='plotly_white',
    )

    # Save
    Out_Fi = Path(Out_Fi)
    if not Out_Fi.name.endswith('.html'):
        Out_Fi = Out_Fi.with_suffix('.html')

    # Save to first path if multiple paths provided
    save_path = l_Pa[0] / Out_Fi
    fig.write_html(save_path)
    print(f'Plot saved to: {save_path}')

    # Show
    fig.show()
