# components/kpi.py
def kpi_card(label: str, value: str, delta: str | None = None):
    delta_html = f"<span class='kpi-delta'>{delta}</span>" if delta else ""
    return f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}{delta_html}</div>
    </div>"""
