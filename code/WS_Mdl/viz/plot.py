import re
from datetime import datetime as DT
from pathlib import Path

import plotly.graph_objects as go

from WS_Mdl.core import vprint


def p_TS_range(l_Pa, ending='IDF', date_format='%Y%m%d', Out_Fi='TS_range.png'):
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
            vprint(f'Warning: Path does not exist: {Pa}')
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
                    vprint(f'Warning: Could not parse date from filename: {f}')
            else:
                vprint(f'Warning: No date pattern found in filename: {f}')

        if l_Dt:
            l_Dt.sort()
            data_by_path[Pa] = l_Dt
            vprint(
                f'Found {len(l_Dt)} files with dates in {Pa.name} ranging from {l_Dt[0].date()} to {l_Dt[-1].date()}'
            )
        else:
            vprint(f'No valid dates found in {Pa}')

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
