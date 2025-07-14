from flask import Flask, render_template, request
import pandas as pd
import joblib
from io import StringIO
import json

app = Flask(__name__)
model = joblib.load("pipeline_rf.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']

    try:
        # Try reading CSV
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            return render_template('index.html', error="Error: The uploaded file is empty.")
        except pd.errors.ParserError:
            return render_template('index.html', error="Error: Failed to parse CSV. Please check formatting.")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin1')

        if df.empty or df.shape[1] == 0:
            return render_template('index.html', error="Error: The uploaded file has no valid columns.")

        # Make predictions
        preds = model.predict(df)
        df['Prediction'] = preds

        # 🔄 Replace 'unknown' with 'normal' in displayed predictions
        df['Prediction'] = df['Prediction'].replace('unknown', 'normal')

        # Chart data for visualization
        chart_data = df['Prediction'].value_counts().to_dict()

        # Table to show selected columns (if exist)
        display_columns = ['srcip', 'dstip', 'sport', 'dsport', 'proto']
        table_columns = [col for col in display_columns if col in df.columns] + ['Prediction']
        table_html = df[table_columns].to_html(classes='result', index=False)

        # Summary values (optional)
        total = len(df)
        attacks = (df['Prediction'] != 'normal').sum()
        normal = (df['Prediction'] == 'normal').sum()
        rate = round((attacks / total) * 100, 2) if total > 0 else 0

        return render_template(
            'result.html',
            tables=[table_html],
            chart_data=json.dumps(chart_data),
            total=total,
            attacks=attacks,
            normal=normal,
            rate=rate
        )

    except Exception as e:
        return render_template('index.html', error=f"Error during prediction: {e}")

if __name__ == '__main__':
    app.run(debug=True)
