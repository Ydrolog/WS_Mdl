# helper file for PoP_SFR_stage_NBr47.ipynb - contains the plotting function for SFR stage vs RIV stage time series for a given reach

import plotly.graph_objects as go


def plot_SFR_reach_TS(r_info, X_axis, SFR_Stg, d_Ct, Pa_Out, HD=None, HD_RIV=None, A_P=None):
    reach_cellid = f'L{r_info["L"]}_R{r_info["R"]}_C{r_info["C"]}'

    sub_title = f'Reach number: {r_info["reach"]}, L: {r_info["L"]}, R: {r_info["R"]}, C: {r_info["C"]}, X: {r_info["X"]}, Y: {r_info["Y"]} - MdlN: {r_info["MdlN"]}, MdlN_RIV: {r_info["MdlN_RIV"]}'

    fig = go.Figure()

    # ---------------------------
    # Trace Addition
    # ---------------------------
    n = len(SFR_Stg)

    # 1. Precipitation (yaxis2 - Top)
    if A_P is not None:
        fig.add_trace(
            go.Bar(
                x=X_axis,
                y=A_P.values if hasattr(A_P, 'values') else A_P,
                name=f'{"Precipitation":<20}',
                marker=dict(color='light blue', line=dict(width=0)),
                hovertemplate='%{y:8.1f} mm',
                yaxis='y2',
            )
        )

    # 2. Elevation Constants (yaxis - Bottom)
    for k in d_Ct.keys():
        fig.add_trace(
            go.Scatter(
                x=X_axis,
                y=[d_Ct[k]['value']] * n,
                mode='lines',
                name=f'{k:<17}',
                line=d_Ct[k]['line'],
                hovertemplate='%{y:3.3f} mNAP',
            )
        )

    # 3. SFR Stage (yaxis - Bottom)
    fig.add_trace(
        go.Scatter(
            x=X_axis,
            y=SFR_Stg[reach_cellid],
            mode='lines',
            name=f'{"SFR stage":<17}',
            line=dict(color='#ff0000', width=3),
            hovertemplate='%{y:3.3f} mNAP',
            showlegend=True,
        )
    )

    # 4. Heads (yaxis - Bottom)
    if HD is not None:
        fig.add_trace(
            go.Scatter(
                x=X_axis,
                y=HD['head'],
                mode='lines',
                name=f'{"Head " + r_info["MdlN"]:<17}',
                line=dict(color='#e26a5a', width=3),
                hovertemplate='%{y:3.3f} mNAP',
                showlegend=True,
            )
        )

    if HD_RIV is not None:
        fig.add_trace(
            go.Scatter(
                x=X_axis,
                y=HD_RIV['head'],
                mode='lines',
                name=f'{"Head " + r_info["MdlN_RIV"]:<17}',
                line=dict(color='#6baed6', width=3),
                hovertemplate='%{y:3.3f} mNAP',
                showlegend=True,
            )
        )

    # ---------------------------
    # Auto-Scaling Calculation for Bottom Plot
    # ---------------------------
    y_vals = []
    # SFR Stage
    series = SFR_Stg[reach_cellid]
    y_vals.extend(series[series.notna()].values)
    # Constants
    for k in d_Ct.keys():
        y_vals.append(d_Ct[k]['value'])
    # Heads
    if HD is not None:
        y_vals.extend(HD['head'].dropna().values)
    if HD_RIV is not None:
        y_vals.extend(HD_RIV['head'].dropna().values)

    if y_vals:
        y_min, y_max = min(y_vals), max(y_vals)
        # Add 5% padding
        y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.5
        y_vals_range = [y_min - y_pad, y_max + y_pad]
    else:
        y_vals_range = [0, 1]

    # ---------------------------
    # Layout Updates
    # ---------------------------

    # Sort Legend (Using stripped name to match keys)
    l_legend_order = [
        'Precipitation',
        f'Head {r_info["MdlN"]}',
        f'Head {r_info["MdlN_RIV"]}',
        'SFR stage',
        'SFR riverbed top',
        'RIV stage',
        'RIV bottom',
        'DRN elevation',
        'top',
        'bottom',
    ]

    # Helper to strip padding for comparison
    def get_clean_name(t):
        return t.name.strip() if t.name else ''

    fig.data = sorted(
        fig.data,
        key=lambda t: l_legend_order.index(get_clean_name(t))
        if get_clean_name(t) in l_legend_order
        else len(l_legend_order),
    )

    # Using 'domains' to separate plots visually on a single figure
    # Top plot: yaxis2 [0.72, 1]
    # Bottom plot: yaxis [0, 0.68]

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
