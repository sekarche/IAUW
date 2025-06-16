
from flask import Flask, render_template_string, send_from_directory
import pandas as pd
import os
import psutil
from datetime import datetime

app = Flask(__name__)

TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Underwater Enhancement Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <meta http-equiv="refresh" content="15">
    <style>
        body { font-family: Arial; margin: 40px; background: #f9f9f9; }
        h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { padding: 8px 12px; border: 1px solid #ccc; text-align: center; }
        th { background: #333; color: white; }
        tr:nth-child(even) { background: #eee; }
        img { width: 120px; height: auto; }
        canvas { margin-top: 20px; }
        .metrics { margin-bottom: 20px; }
    </style>
</head>
<body>
    <h2>Underwater Enhancement Processing Log</h2>
    <div class="metrics">
        <h3>System Metrics</h3>
        <p><strong>CPU Temperature:</strong> {{ cpu_temp }} °C &nbsp; | &nbsp; <strong>CPU Usage:</strong> {{ cpu_usage }}%</p>
        <p><strong>Last Updated:</strong> {{ timestamp }}</p>
    </div>
    <canvas id="scoreChart" width="600" height="200"></canvas>
    <script>
        const ctx = document.getElementById('scoreChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: {{ timestamps|safe }},
                datasets: [{
                    label: 'Quality Score',
                    data: {{ scores|safe }},
                    fill: false,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.2
                }]
            },
            options: {
                scales: {
                    y: { suggestedMin: 0, suggestedMax: 1 }
                }
            }
        });
    </script>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Input Image</th>
            <th>Output Image</th>
            <th>Quality Score</th>
            <th>Degradation</th>
            <th>Enhancement</th>
            <th>Processing Time (s)</th>
            <th>SSIM</th>
            <th>PSNR</th>
        </tr>
        {% for row in rows %}
        <tr>
            <td>{{ row.timestamp }}</td>
            <td><img src="/file/{{ row.input_image }}" alt="Input"></td>
            <td><img src="/file/{{ row.enhanced_image }}" alt="Output"></td>
            <td>{{ row.quality_score }}</td>
            <td>{{ row.degradation_type }}</td>
            <td>{{ row.enhancement_method }}</td>
            <td>{{ row.processing_time_sec }}</td>
            <td>{{ row.ssim if row.ssim else "-" }}</td>
            <td>{{ row.psnr if row.psnr else "-" }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
'''

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return round(int(f.read()) / 1000.0, 1)
    except:
        return "N/A"

@app.route("/")
def dashboard():
    try:
        df = pd.read_csv("results.csv")
    except:
        df = pd.DataFrame(columns=["timestamp", "input_image", "enhanced_image", "quality_score", "degradation_type", "enhancement_method", "processing_time_sec", "ssim", "psnr"])

    df = df.tail(20)
    timestamps = df["timestamp"].tolist()
    scores = df["quality_score"].tolist()
    rows = df.to_dict(orient="records")
    return render_template_string(
        TEMPLATE,
        timestamps=timestamps,
        scores=scores,
        rows=rows,
        cpu_temp=get_cpu_temp(),
        cpu_usage=psutil.cpu_percent(),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.route('/file/<path:filename>')
def serve_file(filename):
    for folder in ["input_images", "output_images"]:
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            return send_from_directory(folder, filename)
    return "File not found", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
