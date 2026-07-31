def generate_execution_chart(setup):
    img_buf = io.BytesIO()
    geom_calc = setup['geometry_data']
    chart_df = geom_calc['df'].tail(80).copy()
    
    upper_series = pd.Series(geom_calc['upper_line'][-len(chart_df):], index=chart_df.index)
    middle_series = pd.Series(geom_calc['middle_line'][-len(chart_df):], index=chart_df.index)
    lower_series = pd.Series(geom_calc['lower_line'][-len(chart_df):], index=chart_df.index)
    
    mc = mpf.make_marketcolors(up='#089981', down='#f23645', edge='inherit', wick='inherit')
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#2a2e39', y_on_right=True, facecolor='#131722', figcolor='#131722')
    
    addplots = [
        mpf.make_addplot(chart_df['EMA50'], color='#2962ff', width=1.5),
        mpf.make_addplot(upper_series, color='#00e676', width=2.0, linestyle='-'),
        mpf.make_addplot(middle_series, color='#ff9800', width=1.5, linestyle='--'),
        mpf.make_addplot(lower_series, color='#00e676', width=2.0, linestyle='-')
    ]
    
    # Fix compression: Base the y-axis strictly on the visible candles and active price action levels 
    # rather than forcing distant take-profit targets into the viewport scale.
    price_min = chart_df['Low'].min()
    price_max = chart_df['High'].max()
    padding = (price_max - price_min) * 0.15
    if padding == 0:
        padding = 0.001
        
    ymin = price_min - padding
    ymax = price_max + padding
        
    fig, axlist = mpf.plot(
        chart_df, type='candle', style=style, volume=False,
        addplot=addplots, returnfig=True, figsize=(12, 7),
        ylim=(ymin, ymax)
    )
    
    ax = axlist[0]
    ax.axhline(geom_calc['resistance_level'], color='#ffeb3b', linestyle='-', linewidth=1.5, label='Structural Resistance / Neckline')
    ax.axhline(geom_calc['support_level'], color='#ffeb3b', linestyle='-', linewidth=1.5, label='Breaker Block / Support')
    
    # Plot target lines only if they fall within or close to the expanded viewport, or use annotations
    ax.axhline(setup['entry'], color='#00e676', linestyle='--', linewidth=1.2, label='Entry')
    if setup['tp1'] >= ymin and setup['tp1'] <= ymax:
        ax.axhline(setup['tp1'], color='#2962ff', linestyle='--', linewidth=1.2, label='TP1')
    if setup['tp2'] >= ymin and setup['tp2'] <= ymax:
        ax.axhline(setup['tp2'], color='#e53935', linestyle='--', linewidth=1.2, label='TP2')
    
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S WAT")
    ax.set_title(f"{setup['symbol']} Institutional Geometry Map\nPattern: {setup['pattern_name']} | Confidence: {setup['confidence']:.1f}% | {current_time_str}", color='white', fontsize=10, fontweight='bold', pad=12)
    
    fig.savefig(img_buf, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    img_buf.seek(0)
    plt.close(fig)
    return img_buf
