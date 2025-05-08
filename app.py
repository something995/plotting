from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Or 'module://backend_agg'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'dta'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Dummy user for demonstration
users = {'user': 'password'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_plot(data, x_col, y_col, z_col=None, plot_type='scatter'):
    """Generates a Matplotlib plot as a base64 encoded image."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d' if z_col else None)

    try:
        x_data = data[x_col]
        y_data = data[y_col]
        z_data = data[z_col] if z_col else None

        if plot_type == 'scatter':
            if z_data is not None:
                ax.scatter(x_data, y_data, z_data)
                ax.set_zlabel(z_col)
            else:
                ax.scatter(x_data, y_data)
        elif plot_type == 'line':
            if z_data is not None:
                ax.plot(x_data, y_data, z_data)
                ax.set_zlabel(z_col)
            else:
                ax.plot(x_data, y_data)
        elif plot_type == 'bar':
            ax.bar(x_data, y_data)
        elif plot_type == 'hist':
            ax.hist(x_data)
        elif plot_type == 'box':
            ax.boxplot(data[[y_col, x_col]].values)
            ax.set_xticklabels([y_col, x_col])
        elif plot_type == 'surface' and z_data is not None:
            try:
                X, Y = np.meshgrid(np.unique(x_data), np.unique(y_data))
                Z = data.pivot_table(values=z_col, index=y_col, columns=x_col).values
                ax.plot_surface(X, Y, Z)
                ax.set_zlabel(z_col)
            except Exception as e:
                return f"Error creating surface plot: {e}"
        elif plot_type == 'bar3d' and z_data is not None:
            xpos, ypos = np.meshgrid(np.arange(len(np.unique(x_data))), np.arange(len(np.unique(y_data))))
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros_like(xpos)
            dx = dy = 0.5
            dz = data.pivot_table(values=z_col, index=y_col, columns=x_col).values.flatten()
            ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
            ax.set_zlabel(z_col)
            ax.set_xticks(np.arange(len(np.unique(x_data))))
            ax.set_xticklabels(np.unique(x_data))
            ax.set_yticks(np.arange(len(np.unique(y_data))))
            ax.set_yticklabels(np.unique(y_data))
        elif plot_type == 'contour' and z_data is not None:
            try:
                X, Y = np.meshgrid(np.unique(x_data), np.unique(y_data))
                Z = data.pivot_table(values=z_col, index=y_col, columns=x_col).values
                ax.contour3D(X, Y, Z, 50, cmap='binary')
                ax.set_zlabel(z_col)
            except Exception as e:
                return f"Error creating contour plot: {e}"

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.title(f'{plot_type.capitalize()} Plot')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        return plot_url

    except KeyError:
        return "Error: One or more selected columns not found."
    except Exception as e:
        return f"Error generating plot: {e}"

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            return redirect(url_for('upload'))
        else:
            error = 'Invalid credentials'
    return render_template('login.html', error=error)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filename)
                elif filename.endswith('.dta'):
                    df = pd.read_stata(filename)
                else:
                    return "Error: Unsupported file type."
                return render_template('analysis.html', columns=df.columns.tolist())
            except Exception as e:
                return f"Error reading file: {e}"
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    x_axis = request.form.get('x_axis')
    y_axis = request.form.get('y_axis')
    z_axis = request.form.get('z_axis')
    plot_type = request.form.get('plot_type')

    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    if not files:
        return render_template('analysis.html', error="No data file uploaded yet.")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], files[-1])
    try:
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith('.dta'):
            data = pd.read_stata(filepath)
        else:
            return render_template('analysis.html', error="Error: Unsupported file type.")

        plot_url = generate_plot(data, x_axis, y_axis, z_axis, plot_type)
        return render_template('analysis.html', columns=data.columns.tolist(), plot_url=plot_url)

    except Exception as e:
        return render_template('analysis.html', error=f"Error processing data: {e}")

if __name__ == '__main__':
    app.run(debug=True)