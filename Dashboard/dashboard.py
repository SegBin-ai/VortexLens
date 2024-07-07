import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import os

# Function to parse the .obj file
def parse_obj_file(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()
                face = [int(idx.split('/')[0]) - 1 for idx in parts[1:]]
                faces.append(face)
    return vertices, faces

# Path to the .obj file
obj_file_path = r'C:\Users\Aaditya Voruganti\Desktop\VortexLens\Dashboard\model.obj'
vertices, faces = parse_obj_file(obj_file_path)
x, y, z = zip(*vertices)
i, j, k = zip(*[(face[0], face[1], face[2]) for face in faces])

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the Dash app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("3D Model Viewer"), className="text-center")
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='3d-model', config={'scrollZoom': True}),
        ])
    ]),
], fluid=True)

# Callback to update the 3D model
@app.callback(
    Output('3d-model', 'figure'),
    [Input('3d-model', 'hoverData')]
)
def update_model(hoverData):
    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        name='3D Model',
        showscale=True
    ))

    fig.update_layout(
        title="3D Model",
        scene=dict(
            xaxis=dict(nticks=10, range=[min(x), max(x)]),
            yaxis=dict(nticks=10, range=[min(y), max(y)]),
            zaxis=dict(nticks=10, range=[min(z), max(z)]),
        ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10)
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
